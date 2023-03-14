add_mlir_library(MaseTransform
  Synthesis.cpp
  EmitHLS.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  "${MASE_MAIN_INCLUDE_DIR}/mase/mlir/Transforms"  

  DEPENDS
  MaseTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRSupport
  MLIRAffineToStandard
  MLIRLinalgTransforms
  MLIRLinalgUtils
)
