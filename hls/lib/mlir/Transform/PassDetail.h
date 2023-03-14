//===- PassDetail.h -------------------------------------------------------===//
//
// this file includes the transforms pass class details.
//
//===----------------------------------------------------------------------===//

#ifndef MASE_MLIR_TRANSFORMS_PASSDETAIL_H_
#define MASE_MLIR_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mase {
#define GEN_PASS_CLASSES
#include "mase/mlir/Transform/Passes.h.inc"
} // namespace mase

#endif // TRANSFORMS_PASSDETAIL_H_
