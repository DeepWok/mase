//===- Synthesis.cpp ------------------------------------------------------===//
//
// This file implements passes that preprocess the mlir code from Torch-MLIR for
// HLS code generation.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mase/mlir/Transform/MaseTransforms.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <deque>

using namespace mlir;
using namespace llvm;
using namespace mase;

#define DEBUG_TYPE "mase-synthesis"

//----------------------------------------------------------------------------//
// PreprocessFuncPass:
// * The default function emitted from Torch-MLIR has a default function name of
//   "forward". This pass re-names the function based on the user input.
// * The default function contains tensor arguments and results. This pass
//   translates these tensors to memref.
//----------------------------------------------------------------------------//

static LogicalResult preprocessFuncOp(func::FuncOp funcOp, StringRef funcName,
                                      MLIRContext *ctx, ModuleOp moduleOp) {

  LLVM_DEBUG(dbgs() << "Renaming function " << funcOp.getName() << " to "
                    << funcName << "\n");

  // Collect the function arguments and the resturn values for the input
  // function.
  llvm::SmallVector<BlockArgument, 8> oldArguments;
  for (auto &arg : funcOp.getArguments())
    oldArguments.push_back(arg);
  llvm::SmallVector<Value, 8> oldResults;
  funcOp.walk([&](func::ReturnOp op) {
    for (auto res : op.getOperands())
      oldResults.push_back(res);
  });
  LLVM_DEBUG({
    for (auto arg : oldArguments)
      dbgs() << "Found func argument: \n" << arg.getArgNumber() << "\n";
    for (auto res : oldResults)
      dbgs() << "Found func result: \n" << *(res.getDefiningOp()) << "\n";
  });
  if (oldResults.size() != funcOp.getResultTypes().size())
    funcOp.emitError("must have only one return op.");

  // ------------------------------------------------------
  // Preprocess check
  // ------------------------------------------------------

  // Each function argument or result must have tensor type.
  for (auto &argType : funcOp.getArgumentTypes())
    if (!isa<TensorType>(argType))
      funcOp.emitError("Function arguments must be all non-tensor types");
  for (auto resType : funcOp.getResultTypes())
    if (!isa<TensorType>(resType))
      funcOp.emitError("Function results must be all non-tensor types");

  // Each to_memref operation must take either an function argument or
  // a constant in tensor type.
  llvm::DenseMap<Value, bufferization::ToMemrefOp> argumentMap;
  funcOp.walk([&](bufferization::ToMemrefOp op) {
    LLVM_DEBUG(dbgs() << "Found to_memref op: " << op << "\n");
    auto tensorOp = op.getTensor();
    if (std::find(oldArguments.begin(), oldArguments.end(), tensorOp) ==
            oldArguments.end() &&
        !isa<arith::ConstantOp>(tensorOp.getDefiningOp()))
      tensorOp.getDefiningOp()->emitError(
          "ToMemrefOp must take a constant or a block argument.");
    argumentMap[tensorOp] = op;
  });
  llvm::DenseMap<bufferization::ToTensorOp, memref::AllocOp> resultMap;
  funcOp.walk([&](bufferization::ToTensorOp op) {
    llvm::errs() << "Found to_tensor op: " << op << "\n";
    if (std::find(oldResults.begin(), oldResults.end(), op.getResult()) ==
        oldResults.end())
      op.emitError("ToTensorOp must be a returned value.");
    auto memrefOp = dyn_cast<memref::AllocOp>(op.getMemref().getDefiningOp());
    if (!memrefOp)
      op.emitError("Patten mismatch: expect to have alloc memref and "
                   "memref to tensor for return op.");
    resultMap[op] = memrefOp;
  });

  // ------------------------------------------------------
  // Create new FuncOp
  // ------------------------------------------------------

  // Only retain those attributes that are not constructed by build.
  SmallVector<NamedAttribute, 4> attributes;
  for (const auto &attr : funcOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == funcOp.getFunctionTypeAttrName())
      continue;
    attributes.push_back(attr);
  }

  // Get function arguments
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto &argument : argumentMap)
    argTypes.push_back(argument.second.getType());
  for (auto &result : resultMap)
    argTypes.push_back(result.second.getType());

  // Get function results
  llvm::SmallVector<mlir::Type, 8> resTypes;

  // Create a new funcOp of memref type
  ConversionPatternRewriter rewriter(ctx);
  rewriter.setInsertionPoint(funcOp);
  auto func_type = rewriter.getFunctionType(argTypes, resTypes);
  auto newFuncOp = rewriter.create<func::FuncOp>(funcOp.getLoc(), funcName,
                                                 func_type, attributes);

  // Update memref usage
  auto numArgs = funcOp.getBody().getNumArguments();
  auto loc = funcOp.getBody().getArgument(0).getLoc();
  auto i = 0;
  for (auto &argument : argumentMap) {
    auto newArg = funcOp.getBody().addArgument(argument.second.getType(), loc);
    argument.second.replaceAllUsesWith(newArg);
    i++;
  }
  for (auto &result : resultMap) {
    auto newArg = funcOp.getBody().addArgument(result.second.getType(), loc);
    result.second.replaceAllUsesWith(newArg);
  }

  // Clear ReturnOp Operands
  funcOp.walk([&](func::ReturnOp returnOp) {
    rewriter.setInsertionPoint(returnOp);
    rewriter.create<func::ReturnOp>(returnOp.getLoc());
    returnOp.erase();
  });

  // Erase Tensor Ops
  for (auto &argument : argumentMap) {
    auto tensorOp = argument.first;
    assert(argument.second.use_empty());
    argument.second->erase();
    if (auto op = tensorOp.getDefiningOp())
      op->erase();
  }
  for (auto &result : resultMap) {
    assert(result.first.use_empty());
    result.first->erase();
    assert(result.second.use_empty());
    result.second->erase();
  }
  for (unsigned int i = 0; i < numArgs; i++)
    funcOp.getBody().eraseArgument(0);

  // Inline the function body
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  funcOp.erase();

  LLVM_DEBUG(dbgs() << *newFuncOp << "\n");
  return success();
}

namespace {
class PreprocessFuncPass
    : public mase::PreprocessFuncPassBase<PreprocessFuncPass> {

public:
  void runOnOperation() override {

    ModuleOp m = getOperation();

    // Torch-mlir must only emit a single function as the top-level
    // function.
    auto i = 0;
    m.walk([&](func::FuncOp op) { i++; });
    if (i != 1)
      m.emitError("Found more than one function in the module. Please "
                  "check which one for lowering.");

    for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>())) {
      if (failed(preprocessFuncOp(funcOp, funcName, &getContext(), m)))
        return signalPassFailure();
    }

    // Remove ml_program global seed
    for (auto globalOp :
         llvm::make_early_inc_range(m.getOps<ml_program::GlobalOp>()))
      globalOp->erase();
  }
};
} // namespace

std::unique_ptr<Pass> mase::createPreprocessFuncPass() {
  return std::make_unique<PreprocessFuncPass>();
}

//----------------------------------------------------------------------------//
// NameArgumentPass:
//   The MLIR function does not contain named arguments. This pass sets the
//   function arguments' names so it helps the code generation in the later
//   stage.
//----------------------------------------------------------------------------//
