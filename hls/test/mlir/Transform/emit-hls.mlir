// RUN: mase-opt %s --emit-hls="hls-param=data_in,,in,,fixed,,(1,100),,(8,5);data_out,,out,,fixed,,(1,100),,(8,5);" | FileCheck %s

// CHECK: // =====================================
// CHECK-NEXT: //     Mase HLS Hardware
// CHECK-NEXT: //     Model: relu
// CHECK-NEXT: // =====================================
// CHECK-NEXT: #include <algorithm>
// CHECK-NEXT: #include <ap_axi_sdata.h>
// CHECK-NEXT: #include <ap_fixed.h>
// CHECK-NEXT: #include <ap_int.h>
// CHECK-NEXT: #include <hls_math.h>
// CHECK-NEXT: #include <hls_stream.h>
// CHECK-NEXT: #include <math.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <string.h>
// CHECK-NEXT: using namespace std;
// CHECK: void relu(ap_fixed<8, 5> v0[1][100],ap_fixed<8, 5> v1[1][100]) {
// CHECK-NEXT: ap_fixed<8, 5> v2 = 0.000000;
// CHECK-NEXT: ap_fixed<8, 5> v3;
// CHECK-NEXT: bool v4;
// CHECK-NEXT: ap_fixed<8, 5> v5;
// CHECK-NEXT: b0:
// CHECK-NEXT: for ( v6 = 0; v6 < 1; v6 += 1) {b1:
// CHECK-NEXT: for ( v7 = 0; v7 < 100; v7 += 1) {b2:
// CHECK-NEXT: v3 = v0[0][v7];
// CHECK-NEXT: v4 = v3 > v2;
// CHECK-NEXT: v5 = v4 ? v3 : v2;
// CHECK-NEXT: v1[v6][v7] = v5;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @relu(%arg0: memref<1x100xf32>, %arg1: memref<1x100xf32>) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     affine.for %arg2 = 0 to 1 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 100 {
// CHECK-NEXT:         %0 = affine.load %arg0[0, %arg3] : memref<1x100xf32>
// CHECK-NEXT:         %1 = arith.cmpf ugt, %0, %cst : f32
// CHECK-NEXT:         %2 = arith.select %1, %0, %cst : f32
// CHECK-NEXT:         affine.store %2, %arg1[%arg2, %arg3] : memref<1x100xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @relu(%arg0: memref<1x100xf32>, %arg1: memref<1x100xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 100 {
      %0 = affine.load %arg0[0, %arg3] : memref<1x100xf32>
      %1 = arith.cmpf ugt, %0, %cst : f32
      %2 = arith.select %1, %0, %cst : f32
      affine.store %2, %arg1[%arg2, %arg3] : memref<1x100xf32>
    }
  }
  return
}

