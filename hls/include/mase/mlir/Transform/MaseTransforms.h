//===- MaseTransforms.h ----------------------------------------*- C++ -*-===//
//
// This file declares all the registration interfaces for Mase passes.
//
//===----------------------------------------------------------------------===//

#ifndef MASE_MLIR_TRANSFORMS_MASETRANSFORMS_H
#define MASE_MLIR_TRANSFORMS_MASETRANSFORMS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mase {

std::unique_ptr<mlir::Pass> createNameFuncPass();

} // namespace mase

#endif
