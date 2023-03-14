//===- mase-opt.cpp - The mase-opt driver -------------------------------===//
//
// This file implements the 'mase-opt' tool, which is the mase analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mase/mlir/Transform/MaseTransforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/PrettyStackTrace.h"

using namespace llvm;
using namespace mlir;
using namespace mase;

int main(int argc, char *argv[]) {
  DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::emitc::EmitCDialect>();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();

  // Register Mase specific passes.
  registerMasePasses();

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();

  return failed(MlirOptMain(argc, argv, "Mase HLS driver", registry,
                            /*preloadDialectsInContext=*/true));
}
