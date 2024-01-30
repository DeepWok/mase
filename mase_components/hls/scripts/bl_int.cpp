// --- HLS implementation of LLM.int

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
using namespace std;

typedef ap_int<16> HTYPE;
typedef ap_int<4> LTYPE;
#define X_ROW 1
#define X_COL 4096
#define W_ROW X_COL
#define W_COL 11008
#define A_ROW X_COL
#define A_COL 32
#define B_ROW A_COL
#define B_COL W_COL

void mm1(HTYPE x[X_ROW][X_COL], LTYPE w[W_ROW][W_COL], HTYPE x1[X_ROW][W_COL]) {
#pragma HLS array_partition variable = x type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = w type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = x1 type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF

loop_x_row:
  for (int i = 0; i < X_ROW; i++) {
  loop_w:
    for (int k = 0; k < W_COL; k++) {
      HTYPE res = 0;
    loop_x_col:
      for (int j = 0; j < X_COL; j += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        res += x[i][j] * w[j][k] + x[i][j + 1] * w[j + 1][k] +
               x[i][j + 2] * w[j + 2][k] + x[i][j + 3] * w[j + 3][k] +
               x[i][j + 4] * w[j + 4][k] + x[i][j + 5] * w[j + 5][k] +
               x[i][j + 6] * w[j + 6][k] + x[i][j + 7] * w[j + 7][k] +
               x[i][j + 8] * w[j + 8][k] + x[i][j + 9] * w[j + 9][k] +
               x[i][j + 10] * w[j + 10][k] + x[i][j + 11] * w[j + 11][k] +
               x[i][j + 12] * w[j + 12][k] + x[i][j + 13] * w[j + 13][k] +
               x[i][j + 14] * w[j + 14][k] + x[i][j + 15] * w[j + 15][k];
#pragma HLS bind_op variable = res op = add impl = fabric
#pragma HLS bind_op variable = res op = mul impl = fabric
      }
      x1[i][k] = res;
    }
  }
}

void mm2(HTYPE x[X_ROW][X_COL], HTYPE a[A_ROW][A_COL], HTYPE xa[X_ROW][A_COL]) {
#pragma HLS array_partition variable = x type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = a type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = xa type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF

loop_x_row:
  for (int i = 0; i < X_ROW; i++) {
  loop_w:
    for (int k = 0; k < A_COL; k++) {
      HTYPE res = 0;
    loop_x_col:
      for (int j = 0; j < X_COL; j += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        res += x[i][j] * a[j][k] + x[i][j + 1] * a[j + 1][k] +
               x[i][j + 2] * a[j + 2][k] + x[i][j + 3] * a[j + 3][k] +
               x[i][j + 4] * a[j + 4][k] + x[i][j + 5] * a[j + 5][k] +
               x[i][j + 6] * a[j + 6][k] + x[i][j + 7] * a[j + 7][k] +
               x[i][j + 8] * a[j + 8][k] + x[i][j + 9] * a[j + 9][k] +
               x[i][j + 10] * a[j + 10][k] + x[i][j + 11] * a[j + 11][k] +
               x[i][j + 12] * a[j + 12][k] + x[i][j + 13] * a[j + 13][k] +
               x[i][j + 14] * a[j + 14][k] + x[i][j + 15] * a[j + 15][k];
#pragma HLS bind_op variable = res op = add impl = fabric
#pragma HLS bind_op variable = res op = mul impl = fabric
      }
      xa[i][k] = res;
    }
  }
}

void mm3(HTYPE xa[X_ROW][A_COL], HTYPE b[B_ROW][B_COL],
         HTYPE x2[X_ROW][B_COL]) {
#pragma HLS array_partition variable = xa type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = b type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = x2 type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF

loop_x_row:
  for (int i = 0; i < X_ROW; i++) {
  loop_w:
    for (int k = 0; k < A_COL; k++) {
      HTYPE res = 0;
    loop_x_col:
      for (int j = 0; j < X_COL; j += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        res += xa[i][j] * b[j][k] + xa[i][j + 1] * b[j + 1][k] +
               xa[i][j + 2] * b[j + 2][k] + xa[i][j + 3] * b[j + 3][k] +
               xa[i][j + 4] * b[j + 4][k] + xa[i][j + 5] * b[j + 5][k] +
               xa[i][j + 6] * b[j + 6][k] + xa[i][j + 7] * b[j + 7][k] +
               xa[i][j + 8] * b[j + 8][k] + xa[i][j + 9] * b[j + 9][k] +
               xa[i][j + 10] * b[j + 10][k] + xa[i][j + 11] * b[j + 11][k] +
               xa[i][j + 12] * b[j + 12][k] + xa[i][j + 13] * b[j + 13][k] +
               xa[i][j + 14] * b[j + 14][k] + xa[i][j + 15] * b[j + 15][k];
#pragma HLS bind_op variable = res op = add impl = fabric
#pragma HLS bind_op variable = res op = mul impl = fabric
      }
      x2[i][k] = res;
    }
  }
}

void add(HTYPE x1[X_ROW][W_COL], HTYPE x2[X_ROW][W_COL],
         HTYPE y[X_ROW][W_COL]) {
#pragma HLS array_partition variable = x1 type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = x2 type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = y type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF

  for (int i = 0; i < X_ROW; i++) {
#pragma HLS LOOP_FLATTEN
    for (int k = 0; k < W_COL; k += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
      y[i][k] = x1[i][k] + x2[i][k];
      y[i][k + 1] = x1[i][k + 1] + x2[i][k + 1];
      y[i][k + 2] = x1[i][k + 2] + x2[i][k + 2];
      y[i][k + 3] = x1[i][k + 3] + x2[i][k + 3];
      y[i][k + 4] = x1[i][k + 4] + x2[i][k + 4];
      y[i][k + 5] = x1[i][k + 5] + x2[i][k + 5];
      y[i][k + 6] = x1[i][k + 6] + x2[i][k + 6];
      y[i][k + 7] = x1[i][k + 7] + x2[i][k + 7];
      y[i][k + 8] = x1[i][k + 8] + x2[i][k + 8];
      y[i][k + 9] = x1[i][k + 9] + x2[i][k + 9];
      y[i][k + 10] = x1[i][k + 10] + x2[i][k + 10];
      y[i][k + 11] = x1[i][k + 11] + x2[i][k + 11];
      y[i][k + 12] = x1[i][k + 12] + x2[i][k + 12];
      y[i][k + 13] = x1[i][k + 13] + x2[i][k + 13];
      y[i][k + 14] = x1[i][k + 14] + x2[i][k + 14];
      y[i][k + 15] = x1[i][k + 15] + x2[i][k + 15];
    }
  }
}

void top(HTYPE x[X_ROW][X_COL], LTYPE w[W_ROW][W_COL], HTYPE a[A_ROW][A_COL],
         HTYPE b[B_ROW][B_COL], HTYPE y[X_ROW][W_COL]) {
  HTYPE x1[X_ROW][W_COL];
  HTYPE xa[X_ROW][A_COL];
  HTYPE x2[X_ROW][B_COL];
#pragma HLS array_partition variable = x type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = xa type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = a type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = b type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = w type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = x1 type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = x2 type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = y type = cyclic factor = 16 dim = 2

  mm1(x, w, x1);
  mm2(x, a, xa);
  mm3(xa, b, x2);
  add(x1, x2, y);
}
