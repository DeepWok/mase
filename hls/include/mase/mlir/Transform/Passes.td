#ifndef MASE_MLIR_TRANSFORMS_PASSES
#define MASE_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def NameFuncPass : Pass<"name-func", "::mlir::ModuleOp"> {
  let summary = "Update the name of the function";
  let description = [{
    Update the name of the function.
  }];

  let constructor = "mase::createNameFuncPass()";

  let options = [
    Option<"funcName", "func-name", "std::string", /*default=*/"",
           "The specified name to be updated">
  ];
}

#endif
