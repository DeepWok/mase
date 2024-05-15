//===- MaseTransforms.h -------------------------------------------------===//
//
// this file declares all the registration interfaces for mase passes.
//
//===----------------------------------------------------------------------===//

#ifndef MASE_MLIR_TRANSFORMS_MASETRANSFORMS_H
#define MASE_MLIR_TRANSFORMS_MASETRANSFORMS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mase {

std::unique_ptr<mlir::Pass> createPreprocessFuncPass();
std::unique_ptr<mlir::Pass> createEmitHLSPass();

void registerMasePasses();

} // namespace mase

#endif
