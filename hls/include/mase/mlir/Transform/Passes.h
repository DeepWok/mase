#ifndef MASE_MLIR_TRANSFORMS_PASSES_H
#define MASE_MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include "mase/mlir/Transform/MaseTransforms.h"

namespace mase {

void registerMasePasses();

} // namespace mase

#endif
