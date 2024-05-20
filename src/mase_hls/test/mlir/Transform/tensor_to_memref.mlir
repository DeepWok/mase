// RUN: mase-opt %s --preprocess-func=func-name=relu | FileCheck %s

// CHECK: func.func @relu(%arg0: memref<1x100xf32>, %arg1: memref<1x100xf32>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   affine.for %arg2 = 0 to 1 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 100 {
// CHECK-NEXT:       %0 = affine.load %arg0[0, %arg3] : memref<1x100xf32>
// CHECK-NEXT:       %1 = arith.cmpf ugt, %0, %cst : f32
// CHECK-NEXT:       %2 = arith.select %1, %0, %cst : f32
// CHECK-NEXT:       affine.store %2, %arg1[%arg2, %arg3] : memref<1x100xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @forward(%arg0: tensor<1x100xf32>) -> tensor<1x100xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x100xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x100xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 100 {
      %2 = affine.load %0[0, %arg2] : memref<1x100xf32>
      %3 = arith.cmpf ugt, %2, %cst : f32
      %4 = arith.select %3, %2, %cst : f32
      affine.store %4, %alloc[%arg1, %arg2] : memref<1x100xf32>
    }
  }
  %1 = bufferization.to_tensor %alloc : memref<1x100xf32>
  return %1 : tensor<1x100xf32>
}

