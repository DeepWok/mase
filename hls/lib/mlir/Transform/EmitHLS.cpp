//===- EmitHLS.cpp --------------------------------------------------------===//
//
// This file implements passes that emits HLS code from the given MLIR code.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mase/mlir/Transform/MaseTransforms.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace mase;

#define DEBUG_TYPE "mase-emit-hls"

//----------------------------------------------------------------------------//
// EmitHLSPass:
// * Translates the input MLIR code to HLS code for Vitis HLS.
// * Analysis may be applied to work out whether the input program can be
//   transformed from block interface into handshake interface (also known as
//   dataflow).
//----------------------------------------------------------------------------//

static auto blockID = 0;
static auto valueID = 0;
static llvm::DenseMap<Value, std::string> valueMap;
static llvm::DenseMap<Block *, std::string> blockMap;

// ------------------------------------------------------
//  Inferring variable types
//  The trained hardware module contains quantized precisions for all the data
//  and parameters. In the software mode, these precisions are modelled using
//  floating points, which are inefficient in hardware. Here we load the
//  quantized precision information and directly generate types in the correct
//  formats.
// ------------------------------------------------------

class QArgType {
public:
  std::string name;
  bool isInput;
  std::string type;
  std::vector<long int> shape;
  std::vector<long int> precision;
  std::string typeAsString;
};

static std::string globalType;
// Unit/Element type of a value
static llvm::DenseMap<Value, QArgType *> argTypeMap;

static bool isNumberString(std::string s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

static std::vector<long int> parseVector(std::string param) {
  std::vector<long int> vec;

  size_t pos = 0;
  std::string token;
  while ((pos = param.find(",")) != std::string::npos) {
    token = param.substr(0, pos);
    param.erase(0, pos + 1);
    LLVM_DEBUG({
      if (!isNumberString(token))
        dbgs() << token << "\n";
    });
    assert(isNumberString(token) &&
           "Non digit char is found when parsing vectors");
    vec.push_back(std::stoi(token));
  }
  assert(param.empty() &&
         "Vector is not completely parsed. Missing the last comma?");
  return vec;
}

template <typename OpType>
static bool areVecSame(std::vector<OpType> &v1, std::vector<OpType> &v2) {
  if (v1.size() != v2.size())
    return false;
  return std::equal(v1.begin(), v1.begin() + v1.size(), v2.begin());
}

static int countSubstring(std::string pat, std::string txt) {
  int M = pat.length();
  int N = txt.length();
  int res = 0;

  /* A loop to slide pat[] one by one */
  for (int i = 0; i <= N - M; i++) {
    /* For current index i, check for
       pattern match */
    int j;
    for (j = 0; j < M; j++)
      if (txt[i + j] != pat[j])
        break;

    // if pat[0...M-1] = txt[i, i+1, ...i+M-1]
    if (j == M) {
      res++;
    }
  }
  return res;
}

// Parse the HLS parameters, which is in the format of:
// "{name},{direction},{ty},{size},{precision};"
static std::vector<QArgType *> parseHlsParam(std::string param) {
  std::vector<QArgType *> types;

  size_t pos = 0;
  std::string token;
  while ((pos = param.find(";")) != std::string::npos) {
    token = param.substr(0, pos);
    param.erase(0, pos + 1);

    // Parse a single arg
    if (countSubstring(",,", token) != 4)
      llvm_unreachable(
          "Invalid HLS paramter format (expected 4 commas), please double "
          "check: {name},,{direction},,{ty},,{size},,{precision};");

    auto newTy = new QArgType;
    auto subPos = token.find(",,");
    newTy->name = token.substr(0, subPos);
    token.erase(0, subPos + 2);
    subPos = token.find(",,");
    newTy->isInput = (token.substr(0, subPos) == "in");
    token.erase(0, subPos + 2);
    subPos = token.find(",,");
    newTy->type = (token.substr(0, subPos) == "fixed") ? "ap_fixed" : "unknown";
    token.erase(0, subPos + 2);
    subPos = token.find(",,");
    // Remove the brackets and double comma
    newTy->shape = parseVector(token.substr(1, subPos - 2) + ",");
    token.erase(0, subPos + 2);
    // Remove the brackets, double comma and semicolon
    newTy->precision = parseVector(token.substr(1, token.size() - 2) + ",");
    types.push_back(newTy);
  }
  assert(param.empty() &&
         "Param is not completely parsed. Missing the last semicolon?");

  // Verify: we use types and shapes to find the unnamed arguments. So if there
  // are two args that have the same shape and type, there may be a bug.
  // For now we assume that the input parametes must be in order.
  // for (long unsigned int i = 0; i < types.size(); i++)
  //  for (long unsigned int j = i + 1; j < types.size(); j++)
  //    assert(!areVecSame<long int>(types[i]->shape, types[j]->shape) &&
  //           "Found two args that have the same shape - cannot distinguish "
  //           "between them for quantization");

  return types;
}

static void initArgTypeMap(func::FuncOp funcOp, std::string hlsParam) {
  LLVM_DEBUG(dbgs() << funcOp.getName() << " : " << hlsParam << "\n");

  auto types = parseHlsParam(hlsParam);

  for (auto &arg : funcOp.getArguments()) {
    auto arrayType = dyn_cast<ShapedType>(arg.getType());
    if (!arrayType)
      llvm_unreachable("Function arg has scalar type. This is not supported.");
    if (!arrayType.hasStaticShape())
      llvm_unreachable(
          "Function arg contains dynamic shape. This is not supported.");

    for (auto type : types) {
      std::vector<long int> arrayShape = arrayType.getShape().vec();
      if (areVecSame<long int>(type->shape, arrayShape)) {
        if (type->type == "ap_fixed") {
          assert(type->precision.size() == 2 &&
                 "Function arg has ap_fixed type but has more than 2 precision "
                 "constraints.");
          assert(
              type->precision[0] >= type->precision[1] &&
              "Function arg has ap_fixed type but total width is smaller than "
              "the fraction width.");

          type->typeAsString = type->type + "<" +
                               std::to_string(type->precision[0]) + ", " +
                               std::to_string(type->precision[1]) + ">";
          argTypeMap[arg] = type;
          valueMap[arg] = type->name;
        } else
          llvm_unreachable("Unsupported data type for the function arguments.");

        if (!type->isInput)
          globalType = argTypeMap[arg]->typeAsString;

        // Assuming that the types are in-order
        types.erase(std::find(types.begin(), types.end(), type));
        break;
      }
    }
    assert(argTypeMap.count(arg) &&
           "Cannot find arg with the same shape in the input HLS parameters.");
  }
  assert(types.empty());
}

// ------------------------------------------------------
//   Utils
// ------------------------------------------------------

static std::string getValueName(Value value) {
  if (valueMap.count(value) == 0)
    valueMap[value] = "v" + std::to_string(valueID++);

  return valueMap[value];
}

static std::string getBlockName(Block *block) {
  if (blockMap.count(block) == 0)
    blockMap[block] = "b" + std::to_string(blockID++);

  return blockMap[block];
}

static std::string getTypeName(Value value) {
  if (dyn_cast<BlockArgument>(value) and argTypeMap.count(value) == 1)
    return argTypeMap[value]->typeAsString;

  auto valueType = value.getType();
  if (auto arrayType = dyn_cast<ShapedType>(valueType))
    valueType = arrayType.getElementType();

  if (isa<Float32Type>(valueType))
    return globalType;
  if (isa<Float64Type>(valueType))
    return globalType;

  if (isa<IndexType>(valueType))
    return "int";
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    if (intType.getWidth() == 1)
      return "bool";
    else
      return globalType;
  }
  value.getDefiningOp()->emitError("has unsupported type.");
  return "";
}

static std::string getConstantOperand(Type type, Attribute attr) {

  if (type.isInteger(1))
    return std::to_string(attr.cast<BoolAttr>().getValue());

  std::string constValue;
  if (type.isIndex())
    return std::to_string(attr.cast<IntegerAttr>().getInt());

  if (auto floatType = type.dyn_cast<FloatType>()) {
    auto value = attr.cast<FloatAttr>().getValue().convertToDouble();
    return std::isfinite(value) ? std::to_string(value)
                                : (value > 0 ? "INFINITY" : "-INFINITY");
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.isSigned())
      return std::to_string(attr.cast<IntegerAttr>().getValue().getSExtValue());
    if (intType.isUnsigned())
      return std::to_string(attr.cast<IntegerAttr>().getValue().getZExtValue());
    return std::to_string(attr.cast<IntegerAttr>().getInt());
  }

  llvm::errs() << type << "\n";
  llvm_unreachable("Unsupported type for constant op");
  return constValue;
}

static std::string getArrayInit(Value array) {

  auto arrayType = array.getType().dyn_cast<ShapedType>();

  if (!arrayType.hasStaticShape())
    llvm_unreachable(
        "Function arg contains dynamic shape. This is not supported.");

  std::string arrayBuff;
  if (arrayType.hasStaticShape()) {
    arrayBuff += getValueName(array);
    for (auto &shape : arrayType.getShape())
      arrayBuff += "[" + std::to_string(shape) + "]";
  } else
    // Treat it as a pointer
    arrayBuff += "*" + getValueName(array);

  return arrayBuff;
}

// ------------------------------------------------------
//  Emit general ops
// ------------------------------------------------------

static std::string emitBinaryOp(Operation *op, std::string symbol) {
  return getValueName(op->getResult(0)) + " = " +
         getValueName(op->getOperand(0)) + " " + symbol + " " +
         getValueName(op->getOperand(1)) + ";\n";
}

static std::string emitBinaryFunc(Operation *op, std::string symbol) {
  return getValueName(op->getResult(0)) + " = " + symbol + "(" +
         getValueName(op->getOperand(0)) + ", " +
         getValueName(op->getOperand(1)) + ");\n";
}

static std::string emitUnaryOp(Operation *op, std::string symbol) {
  return getValueName(op->getResult(0)) + " = " + symbol + "(";
  getValueName(op->getOperand(0)) + ");\n";
}

static std::string emitAssignOp(Operation *op) {
  return getValueName(op->getResult(0)) + " = " +
         getValueName(op->getOperand(0)) + ";\n";
}

class AffineExprPrinter : public AffineExprVisitor<AffineExprPrinter> {
public:
  explicit AffineExprPrinter(unsigned numDim, Operation::operand_range operands)
      : numDim(numDim), operands(operands) {}

  void visitAddExpr(AffineBinaryOpExpr expr) { visitAffineBinary(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { visitAffineBinary(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { visitAffineBinary(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    visitAffineBinary(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    buff += "(";
    visit(expr.getLHS());
    buff += " + ";
    visit(expr.getRHS());
    buff += " - 1) / ";
    visit(expr.getRHS());
    buff += ")";
  }
  void visitConstantExpr(AffineConstantExpr expr) {
    buff += std::to_string(expr.getValue());
  }
  void visitDimExpr(AffineDimExpr expr) {
    buff += getValueName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    buff += getValueName(operands[numDim + expr.getPosition()]);
  }
  void visitAffineBinary(AffineBinaryOpExpr expr, std::string symbol) {
    buff += "(";
    if (auto constRHS = expr.getRHS().dyn_cast<AffineConstantExpr>()) {
      if (symbol == "*" && constRHS.getValue() == -1) {
        buff += "-";
        visit(expr.getLHS());
        buff += ")";
      }
      if (symbol == "+" && constRHS.getValue() < 0) {
        buff += "-";
        visit(expr.getLHS());
        buff += std::to_string(-constRHS.getValue()) + ")";
      }
    }
    if (auto binaryRHS = expr.getRHS().dyn_cast<AffineBinaryOpExpr>()) {
      if (auto constRHS = binaryRHS.getRHS().dyn_cast<AffineConstantExpr>()) {
        if (symbol == "+" && constRHS.getValue() == -1 &&
            binaryRHS.getKind() == AffineExprKind::Mul) {
          visit(expr.getLHS());
          buff += " - ";
          visit(binaryRHS.getLHS());
          buff += ")";
        }
      }
    }
    visit(expr.getLHS());
    buff += " " + symbol + " ";
    visit(expr.getRHS());
    buff += ")";
  }

  std::string getAffineExpr(AffineExpr expr) {
    buff.clear();
    visit(expr);
    return buff;
  }

private:
  std::string buff;
  unsigned numDim;
  Operation::operand_range operands;
};

template <typename OpType>
static std::string emitAffineMaxMinOp(OpType op, std::string symbol) {
  std::string affineMaxMinBuff = getValueName(op.getResult()) + " = ";
  auto affineMap = op.getAffineMap();
  AffineExprPrinter affineMapPrinter(affineMap.getNumDims(), op.getOperands());
  for (unsigned i = 0, e = affineMap.getNumResults() - 1; i < e; ++i)
    affineMaxMinBuff += symbol + "(";
  affineMaxMinBuff += affineMapPrinter.getAffineExpr(affineMap.getResult(0));
  for (auto &expr : llvm::drop_begin(affineMap.getResults(), 1))
    affineMaxMinBuff += ", " + affineMapPrinter.getAffineExpr(expr) + ")";
  return affineMaxMinBuff + ";\n";
}

// ------------------------------------------------------
//   Emit data flow ops
// ------------------------------------------------------

static std::string emitOp(memref::LoadOp loadOp) {
  std::string loadBuff = getValueName(loadOp.getResult()) + " = " +
                         getValueName(loadOp.getMemRef());
  for (auto index : loadOp.getIndices())
    loadBuff += "[" + getValueName(index) + "]";

  loadBuff += ";\n";
  return loadBuff;
}

static std::string emitOp(memref::StoreOp storeOp) {
  std::string storeBuff = getValueName(storeOp.getMemRef());
  for (auto index : storeOp.getIndices())
    storeBuff += "[" + getValueName(index) + "]";
  storeBuff += " = " + getValueName(storeOp.getValueToStore()) + ";\n";
  return storeBuff;
}

static std::string emitOp(memref::CopyOp copyOp) {
  auto type = copyOp.getTarget().getType().cast<MemRefType>();
  return "memcpy(" + getValueName(copyOp.getTarget()) + ", " +
         getValueName(copyOp.getSource()) + ", " +
         std::to_string(type.getNumElements()) + " * sizeof(" +
         getTypeName(copyOp.getTarget()) + "));\n";
}

static std::string emitOp(arith::SelectOp selectOp) {
  return getValueName(selectOp.getResult()) + " = " +
         getValueName(selectOp.getCondition()) + " ? " +
         getValueName(selectOp.getTrueValue()) + " : " +
         getValueName(selectOp.getFalseValue()) + ";\n";
}

static std::string emitOp(arith::CmpFOp cmpFOp) {
  switch (cmpFOp.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return emitBinaryOp(cmpFOp, "==");
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return emitBinaryOp(cmpFOp, "!=");
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return emitBinaryOp(cmpFOp, "<");
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return emitBinaryOp(cmpFOp, "<=");
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return emitBinaryOp(cmpFOp, ">");
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return emitBinaryOp(cmpFOp, ">=");
  default:
    cmpFOp.emitError(" has unsupported compare type.");
  }
  return "";
}

static std::string emitOp(arith::CmpIOp cmpIOp) {
  switch (cmpIOp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return emitBinaryOp(cmpIOp, "==");
  case arith::CmpIPredicate::ne:
    return emitBinaryOp(cmpIOp, "!=");
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return emitBinaryOp(cmpIOp, "<");
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return emitBinaryOp(cmpIOp, "<=");
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return emitBinaryOp(cmpIOp, ">");
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return emitBinaryOp(cmpIOp, ">=");
  default:
    cmpIOp.emitError(" has unsupported compare type.");
  }
  return "";
}

// ------------------------------------------------------
//   Emit control flow ops
// ------------------------------------------------------

static std::string emitBlock(Block &block);

static std::string emitOp(scf::ForOp forOp) {
  auto iter = forOp.getInductionVar();
  auto iterName = getValueName(iter);
  auto lowerBound = forOp.getLowerBound();
  auto upperBound = forOp.getUpperBound();
  auto step = forOp.getStep();

  return "for (" + getTypeName(iter) + " " + iterName + " = " +
         getValueName(lowerBound) + "; " + iterName + " < " +
         getValueName(upperBound) + "; " + iterName +
         " += " + getValueName(step) + ") {" + emitBlock(*forOp.getBody()) +
         "}\n";
}

static std::string emitOp(scf::IfOp ifOp) {
  auto cond = ifOp.getCondition();
  std::string ifBuff;

  ifBuff = "if (" + getValueName(cond) + ") {" +
           emitBlock(ifOp.getThenRegion().front());
  if (!ifOp.getElseRegion().empty()) {
    ifBuff += "} else {\n" + emitBlock(ifOp.getElseRegion().front());
  }

  ifBuff += "}\n";
  return ifBuff;
}

static std::string emitOp(scf::YieldOp yieldOp) {
  if (yieldOp.getNumOperands() == 0)
    return "";

  auto resultIdx = 0;
  std::string yieldBuff;
  for (auto result : yieldOp->getParentOp()->getResults())
    yieldBuff += getValueName(result) + " = " +
                 getValueName(yieldOp.getOperand(resultIdx++)) + ";\n";
  return yieldBuff;
}

static std::string emitOp(AffineForOp affineForOp) {
  auto iter = affineForOp.getInductionVar();
  auto iterName = getValueName(iter);
  auto step = affineForOp.getStep();

  std::string lowerBound;
  auto lowerMap = affineForOp.getLowerBoundMap();
  AffineExprPrinter lowerBoundPrinter(lowerMap.getNumDims(),
                                      affineForOp.getLowerBoundOperands());
  if (lowerMap.getNumResults() == 1)
    lowerBound = lowerBoundPrinter.getAffineExpr(lowerMap.getResult(0));
  else {
    for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
      lowerBound += "max(";
    lowerBound += lowerBoundPrinter.getAffineExpr(lowerMap.getResult(0));
    for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1))
      lowerBound += ", " + lowerBoundPrinter.getAffineExpr(expr) + ")";
  }

  std::string upperBound;
  auto upperMap = affineForOp.getUpperBoundMap();
  AffineExprPrinter upperBoundPrinter(upperMap.getNumDims(),
                                      affineForOp.getUpperBoundOperands());
  if (upperMap.getNumResults() == 1)
    upperBound = upperBoundPrinter.getAffineExpr(upperMap.getResult(0));
  else {
    for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
      upperBound += "min(";
    upperBound += upperBoundPrinter.getAffineExpr(upperMap.getResult(0));
    for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1))
      upperBound += ", " + upperBoundPrinter.getAffineExpr(expr) + ")";
  }

  return "for (" + getTypeName(iter) + " " + iterName + " = " + lowerBound +
         "; " + iterName + " < " + upperBound + "; " + iterName +
         " += " + std::to_string(step) + ") {" +
         emitBlock(*affineForOp.getBody()) + "}\n";
}

static std::string emitOp(AffineIfOp affineIfOp) {
  auto constrSet = affineIfOp.getIntegerSet();
  AffineExprPrinter constrPrinter(constrSet.getNumDims(),
                                  affineIfOp.getOperands());
  std::string affineIfBuff = "if (";
  unsigned constrIdx = 0;
  for (auto &expr : constrSet.getConstraints()) {
    affineIfBuff += constrPrinter.getAffineExpr(expr);
    if (constrSet.isEq(constrIdx))
      affineIfBuff += " == 0";
    else
      affineIfBuff += " >= 0";

    if (constrIdx++ != constrSet.getNumConstraints() - 1)
      affineIfBuff += " && ";
  }
  affineIfBuff += ") {" + emitBlock(*affineIfOp.getThenBlock());
  if (affineIfOp.hasElse())
    affineIfBuff += "} else {\n" + emitBlock(*affineIfOp.getElseBlock());
  affineIfBuff += "}\n";
  return affineIfBuff;
}

static std::string emitOp(AffineLoadOp affineLoadOp) {
  std::string affineLoadBuff = getValueName(affineLoadOp.getResult()) + " = " +
                               getValueName(affineLoadOp.getMemRef());
  auto affineMap = affineLoadOp.getAffineMap();
  AffineExprPrinter affineMapPrinter(affineMap.getNumDims(),
                                     affineLoadOp.getMapOperands());
  for (auto index : affineMap.getResults())
    affineLoadBuff += "[" + affineMapPrinter.getAffineExpr(index) + "]";
  return affineLoadBuff + ";\n";
}

static std::string emitOp(AffineStoreOp affineStoreOp) {
  std::string affineStoreBuff = getValueName(affineStoreOp.getMemRef());
  auto affineMap = affineStoreOp.getAffineMap();
  AffineExprPrinter affineMapPrinter(affineMap.getNumDims(),
                                     affineStoreOp.getMapOperands());
  for (auto index : affineMap.getResults()) {
    affineStoreBuff += "[" + affineMapPrinter.getAffineExpr(index) + "]";
  }
  return affineStoreBuff + " = " +
         getValueName(affineStoreOp.getValueToStore()) + ";\n";
}

static std::string emitOp(AffineYieldOp affineYieldOp) {
  if (affineYieldOp.getNumOperands() == 0)
    return "";

  unsigned resultIdx = 0;
  std::string yieldBuff;
  for (auto result : affineYieldOp->getParentOp()->getResults())
    yieldBuff += getValueName(result) + " = " +
                 getValueName(affineYieldOp.getOperand(resultIdx++)) + ";\n";
  return yieldBuff;
}

// ------------------------------------------------------
//   Enumerate ops
// ------------------------------------------------------

static std::string emitBlock(Block &block) {
  std::string blockBuff = getBlockName(&block) + ":\n";

  for (auto &op : block) {

    // Affine Ops
    if (auto castOp = dyn_cast<AffineForOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<AffineIfOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<AffineLoadOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<AffineStoreOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<AffineYieldOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<AffineMaxOp>(op)) {
      blockBuff += emitAffineMaxMinOp<AffineMaxOp>(castOp, "max");
      continue;
    }
    if (auto castOp = dyn_cast<AffineMinOp>(op)) {
      blockBuff += emitAffineMaxMinOp<AffineMinOp>(castOp, "min");
      continue;
    }

    // SCF Ops
    if (auto castOp = dyn_cast<scf::ForOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<scf::IfOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<scf::YieldOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }

    // MemRef Ops
    if (auto castOp = dyn_cast<memref::LoadOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<memref::StoreOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<memref::CopyOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    // Memref init is emitted when declaring the variable
    if (dyn_cast<memref::AllocOp>(op))
      continue;
    if (dyn_cast<memref::AllocaOp>(op))
      continue;

    // Arith Ops
    if (auto castOp = dyn_cast<arith::SelectOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<arith::AddFOp>(op)) {
      blockBuff += emitBinaryOp(&op, "+");
      continue;
    }
    if (auto castOp = dyn_cast<arith::SubFOp>(op)) {
      blockBuff += emitBinaryOp(&op, "-");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MulFOp>(op)) {
      blockBuff += emitBinaryOp(&op, "*");
      continue;
    }
    if (auto castOp = dyn_cast<arith::DivFOp>(op)) {
      blockBuff += emitBinaryOp(&op, "/");
      continue;
    }
    if (auto castOp = dyn_cast<arith::RemFOp>(op)) {
      blockBuff += emitBinaryOp(&op, "%");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MaxFOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "max");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MinFOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "min");
      continue;
    }
    if (auto castOp = dyn_cast<arith::NegFOp>(op)) {
      blockBuff += emitUnaryOp(&op, "-");
      continue;
    }
    if (auto castOp = dyn_cast<arith::AddIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "+");
      continue;
    }
    if (auto castOp = dyn_cast<arith::SubIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "-");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MulIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "*");
      continue;
    }
    if (auto castOp = dyn_cast<arith::DivSIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "/");
      continue;
    }
    if (auto castOp = dyn_cast<arith::RemSIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "%");
      continue;
    }
    if (auto castOp = dyn_cast<arith::DivUIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "/");
      continue;
    }
    if (auto castOp = dyn_cast<arith::RemUIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "%");
      continue;
    }
    if (auto castOp = dyn_cast<arith::XOrIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "^");
      continue;
    }
    if (auto castOp = dyn_cast<arith::AndIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "&");
      continue;
    }
    if (auto castOp = dyn_cast<arith::OrIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "|");
      continue;
    }
    if (auto castOp = dyn_cast<arith::ShLIOp>(op)) {
      blockBuff += emitBinaryOp(&op, "<<");
      continue;
    }
    if (auto castOp = dyn_cast<arith::ShRSIOp>(op)) {
      blockBuff += emitBinaryOp(&op, ">>");
      continue;
    }
    if (auto castOp = dyn_cast<arith::ShRUIOp>(op)) {
      blockBuff += emitBinaryOp(&op, ">>");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MaxSIOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "max");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MinSIOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "min");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MaxUIOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "max");
      continue;
    }
    if (auto castOp = dyn_cast<arith::MinUIOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "min");
      continue;
    }
    // Constants are initialised at declaration
    if (auto castOp = dyn_cast<arith::ConstantOp>(op))
      continue;
    if (auto castOp = dyn_cast<arith::IndexCastOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::UIToFPOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::SIToFPOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::FPToUIOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::FPToSIOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::TruncIOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::TruncFOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::ExtUIOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::ExtSIOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::ExtFOp>(op)) {
      blockBuff += emitAssignOp(&op);
      continue;
    }
    if (auto castOp = dyn_cast<arith::CmpFOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }
    if (auto castOp = dyn_cast<arith::CmpIOp>(op)) {
      blockBuff += emitOp(castOp);
      continue;
    }

    // Math Ops
    if (auto castOp = dyn_cast<math::PowFOp>(op)) {
      blockBuff += emitBinaryFunc(&op, "pow");
      continue;
    }
    if (auto castOp = dyn_cast<math::AbsIOp>(op)) {
      blockBuff += emitUnaryOp(&op, "abs");
      continue;
    }
    if (auto castOp = dyn_cast<math::AbsFOp>(op)) {
      blockBuff += emitUnaryOp(&op, "abs");
      continue;
    }
    if (auto castOp = dyn_cast<math::CeilOp>(op)) {
      blockBuff += emitUnaryOp(&op, "ceil");
      continue;
    }
    if (auto castOp = dyn_cast<math::CosOp>(op)) {
      blockBuff += emitUnaryOp(&op, "cos");
      continue;
    }
    if (auto castOp = dyn_cast<math::SinOp>(op)) {
      blockBuff += emitUnaryOp(&op, "sin");
      continue;
    }
    if (auto castOp = dyn_cast<math::TanhOp>(op)) {
      blockBuff += emitUnaryOp(&op, "tanh");
      continue;
    }
    if (auto castOp = dyn_cast<math::SqrtOp>(op)) {
      blockBuff += emitUnaryOp(&op, "sqrt");
      continue;
    }
    if (auto castOp = dyn_cast<math::RsqrtOp>(op)) {
      blockBuff += emitUnaryOp(&op, "1.0 / sqrt");
    }
    if (auto castOp = dyn_cast<math::ExpOp>(op)) {
      blockBuff += emitUnaryOp(&op, "exp");
      continue;
    }
    if (auto castOp = dyn_cast<math::Exp2Op>(op)) {
      blockBuff += emitUnaryOp(&op, "exp2");
      continue;
    }
    if (auto castOp = dyn_cast<math::LogOp>(op)) {
      blockBuff += emitUnaryOp(&op, "log");
      continue;
    }
    if (auto castOp = dyn_cast<math::Log2Op>(op)) {
      blockBuff += emitUnaryOp(&op, "log2");
      continue;
    }
    if (auto castOp = dyn_cast<math::Log10Op>(op)) {
      blockBuff += emitUnaryOp(&op, "log10");
      continue;
    }

    // A pre-condition for this pass is that the function must not have return
    // value.
    if (auto castOp = dyn_cast<func::ReturnOp>(op))
      continue;

    op.emitError(" is not supported yet.");
  }
  return blockBuff;
}

static void checkFuncOp(func::FuncOp funcOp) {
  if (funcOp.getBlocks().size() != 1)
    funcOp.emitError(" must contain only one block.");
  if (funcOp.getNumResults() != 0)
    funcOp.emitError(" must contain no result.");
  funcOp.walk([&](AffineForOp op) {
    for (auto result : op.getResults())
      if (dyn_cast<ShapedType>(result.getType()))
        op.emitError(" cannot return memref.");
  });
  funcOp.walk([&](scf::ForOp op) {
    for (auto result : op.getResults())
      if (dyn_cast<ShapedType>(result.getType()))
        op.emitError(" cannot return memref.");
  });
  funcOp.walk([&](AffineIfOp op) {
    for (auto result : op.getResults())
      if (dyn_cast<ShapedType>(result.getType()))
        op.emitError(" cannot return memref.");
  });
  funcOp.walk([&](scf::IfOp op) {
    for (auto result : op.getResults())
      if (dyn_cast<ShapedType>(result.getType()))
        op.emitError(" cannot return memref.");
  });
}

static std::string declareValue(Value value) {
  std::string valueBuff;
  auto type = value.getType();

  // If it is a constant op, we initiliase it while declaring the
  // variable.
  if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
    if (dyn_cast<ShapedType>(type)) {
      valueBuff += getTypeName(value) + " " + getArrayInit(value) + " = {";

      auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
      assert(denseAttr);

      for (auto element : denseAttr.getValues<Attribute>()) {
        auto constantValue = getConstantOperand(type, element);
        assert(!constantValue.empty());
        valueBuff += constantValue + ",";
      }
      valueBuff.pop_back();
      valueBuff += "};\n";
      return valueBuff;
    } else
      return getTypeName(value) + " " + getValueName(value) + " = " +
             getConstantOperand(constantOp.getType(), constantOp.getValue()) +
             ";\n";
  }

  if (dyn_cast<ShapedType>(value.getType()))
    return getTypeName(value) + " " + getArrayInit(value) + ";\n";
  return getTypeName(value) + " " + getValueName(value) + ";\n";
}

static std::string emitOp(func::FuncOp funcOp) {
  // Pre-condition check
  checkFuncOp(funcOp);

  // Emit function prototype
  std::string funcOpBuff = "void " + funcOp.getName().str() + "(";

  // Emit input arguments.
  SmallVector<Value, 8> args;
  for (auto &arg : funcOp.getArguments()) {
    if (!dyn_cast<ShapedType>(arg.getType()))
      llvm_unreachable("Function arg has scalar type. This is not supported.");
    auto argName = (dyn_cast<ShapedType>(arg.getType())) ? getArrayInit(arg)
                                                         : getValueName(arg);
    funcOpBuff += getTypeName(arg) + " " + argName + ",";
  }
  // Get rid of the last comma
  funcOpBuff.pop_back();
  funcOpBuff += ") {\n";

  // Collect all the local variables
  funcOp.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (valueMap.count(result) == 1)
        op->emitError(" has been declared.");
      funcOpBuff += declareValue(result);
    }
  });

  // Emit funcOption body.
  funcOpBuff += emitBlock(funcOp.front());
  funcOpBuff += "}\n";
  return funcOpBuff;
}

static LogicalResult emitHLS(func::FuncOp funcOp, raw_ostream &os) {
  os << "// =====================================\n"
     << "//     Mase HLS Hardware\n"
     << "//     Model: " << funcOp.getName() << "\n"
     << "// =====================================\n"
     << "#include <algorithm>\n"
     << "#include <ap_axi_sdata.h>\n"
     << "#include <ap_fixed.h>\n"
     << "#include <ap_int.h>\n"
     << "#include <hls_math.h>\n"
     << "#include <hls_stream.h>\n"
     << "#include <math.h>\n"
     << "#include <stdint.h>\n"
     << "#include <string.h>\n"
     << "using namespace std;\n\n";

  os << emitOp(funcOp);
  return success();
}

namespace {
class EmitHLSPass : public mase::EmitHLSPassBase<EmitHLSPass> {

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

    std::error_code ec;
    llvm::raw_fd_ostream fout(fileName, ec);
    m.walk([&](func::FuncOp funcOp) {
      initArgTypeMap(funcOp, hlsParam);

      auto result = (fileName.empty()) ? emitHLS(funcOp, llvm::outs())
                                       : emitHLS(funcOp, fout);
      if (failed(result))
        return signalPassFailure();
    });
    if (!fileName.empty())
      fout.close();
  }
};
} // namespace

std::unique_ptr<Pass> mase::createEmitHLSPass() {
  return std::make_unique<EmitHLSPass>();
}
