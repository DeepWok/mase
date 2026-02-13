#ifndef MASE_MLIR_TRANSFORMS_PASSES
#define MASE_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PreprocessFuncPass : Pass<"preprocess-func", "::mlir::ModuleOp"> {
  let summary = "Preprocess the function from Torch-MLIR to Mase HLS";
  let description = [{
    * The default function emitted from Torch-MLIR has a default function name of
      "forward". This pass re-names the function based on the user input.
    * The default function contains tensor arguments and results. This pass
      translates these tensors to memref.
  }];

  let constructor = "mase::createPreprocessFuncPass()";

  let options = [
    Option<"funcName", "func-name", "std::string", /*default=*/"",
           "The specified name to be updated">
  ];
}

def EmitHLSPass : Pass<"emit-hls", "::mlir::ModuleOp"> {
  let summary = "Emit HLS code from MLIR";
  let description = [{
      Emit HLS code for the given program. It could be either default implementation
      of traditional HLS hardware, or a dataflow architecture by bufferizing the inputs
      and outputs.
  }];

  let constructor = "mase::createEmitHLSPass()";
  let options = [
    Option<"fileName", "file-name", "std::string", /*default=*/"",
           "The output HLS code">,
    Option<"hlsParam", "hls-param", "std::string", /*default=*/"",
           "The HLS parameters for quantization">
  ];
}

#endif
