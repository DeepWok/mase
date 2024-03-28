//===- Passes.cpp ---------------------------------------------------------===//
//
// this file registers all the mase passes.
//
//===----------------------------------------------------------------------===//

#include "mase/mlir/Transform/MaseTransforms.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "mase/mlir/Transform/Passes.h.inc"
} // namespace

void mase::registerMasePasses() { ::registerMaseTransformPasses(); }
