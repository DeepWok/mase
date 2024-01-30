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

// FP16 uses half, FP32 uses float, FP64 uses double
typedef half FTYPE;
// int4 uses ap_int<4>; int8 uses ap_int<8>
typedef ap_int<4> ITYPE;
#define X_ROW 1
#define X_COL 4096
#define W_ROW X_COL
#define W_COL 11008
#define CW 0.001
#define GROUPS 128
#define GROUP W_COL / GROUPS

void hp_mm(FTYPE xh[X_ROW][X_COL], FTYPE wh[W_ROW][W_COL],
           FTYPE y[X_ROW][W_COL]) {
#pragma HLS array_partition variable = xh type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wh type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = y type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF

loop_x_row:
  for (int i = 0; i < X_ROW; i++) {
  loop_w:
    for (int k = 0; k < W_COL; k++) {
      FTYPE res = 0;
    loop_x_col:
      for (int j = 0; j < X_COL; j += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        res += xh[i][j] * wh[j][k] + xh[i][j + 1] * wh[j + 1][k] +
               xh[i][j + 2] * wh[j + 2][k] + xh[i][j + 3] * wh[j + 3][k] +
               xh[i][j + 4] * wh[j + 4][k] + xh[i][j + 5] * wh[j + 5][k] +
               xh[i][j + 6] * wh[j + 6][k] + xh[i][j + 7] * wh[j + 7][k] +
               xh[i][j + 8] * wh[j + 8][k] + xh[i][j + 9] * wh[j + 9][k] +
               xh[i][j + 10] * wh[j + 10][k] + xh[i][j + 11] * wh[j + 11][k] +
               xh[i][j + 12] * wh[j + 12][k] + xh[i][j + 13] * wh[j + 13][k] +
               xh[i][j + 14] * wh[j + 14][k] + xh[i][j + 15] * wh[j + 15][k];
#pragma HLS bind_op variable = res op = hadd impl = fabric
#pragma HLS bind_op variable = res op = hmul impl = fabric
      }
      y[i][k] = res;
    }
  }
}

void dequant(ITYPE w[W_ROW][W_COL], FTYPE wh[W_ROW][W_COL], FTYPE g[GROUPS]) {
#pragma HLS array_partition variable = w type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = wh type = cyclic factor = 16 dim = 1

  for (int j = 0; j < W_COL; j++) {
    FTYPE gg;
    if (j % GROUP == 0)
      gg = g[j / GROUP];
    for (int i = 0; i < W_ROW; i += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
      auto temp_0 = w[i][j] * (half)CW / 127 * gg;
      auto temp_1 = w[i + 1][j] * (half)CW / 127 * gg;
      auto temp_2 = w[i + 2][j] * (half)CW / 127 * gg;
      auto temp_3 = w[i + 3][j] * (half)CW / 127 * gg;
      auto temp_4 = w[i + 4][j] * (half)CW / 127 * gg;
      auto temp_5 = w[i + 5][j] * (half)CW / 127 * gg;
      auto temp_6 = w[i + 6][j] * (half)CW / 127 * gg;
      auto temp_7 = w[i + 7][j] * (half)CW / 127 * gg;
      auto temp_8 = w[i + 8][j] * (half)CW / 127 * gg;
      auto temp_9 = w[i + 9][j] * (half)CW / 127 * gg;
      auto temp_10 = w[i + 10][j] * (half)CW / 127 * gg;
      auto temp_11 = w[i + 11][j] * (half)CW / 127 * gg;
      auto temp_12 = w[i + 12][j] * (half)CW / 127 * gg;
      auto temp_13 = w[i + 13][j] * (half)CW / 127 * gg;
      auto temp_14 = w[i + 14][j] * (half)CW / 127 * gg;
      auto temp_15 = w[i + 15][j] * (half)CW / 127 * gg;
#pragma HLS bind_op variable = temp_0 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_1 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_2 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_3 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_4 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_5 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_6 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_7 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_8 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_9 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_10 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_11 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_12 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_13 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_14 op = hmul impl = fabric
#pragma HLS bind_op variable = temp_15 op = hmul impl = fabric

      wh[i][j] = temp_0;
      wh[i + 1][j] = temp_1;
      wh[i + 2][j] = temp_2;
      wh[i + 3][j] = temp_3;
      wh[i + 4][j] = temp_4;
      wh[i + 5][j] = temp_5;
      wh[i + 6][j] = temp_6;
      wh[i + 7][j] = temp_7;
      wh[i + 8][j] = temp_8;
      wh[i + 9][j] = temp_9;
      wh[i + 10][j] = temp_10;
      wh[i + 11][j] = temp_11;
      wh[i + 12][j] = temp_12;
      wh[i + 13][j] = temp_13;
      wh[i + 14][j] = temp_14;
      wh[i + 15][j] = temp_15;
    }
  }
}

void top(FTYPE x[X_ROW][X_COL], ITYPE w[W_ROW][W_COL], FTYPE g[GROUPS],
         FTYPE y[X_ROW][W_COL]) {

  FTYPE wh[W_ROW][W_COL];
#pragma HLS array_partition variable = x type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = w type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = wh type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = y type = cyclic factor = 16 dim = 2

  dequant(w, wh, g);
  hp_mm(x, wh, y);
}
