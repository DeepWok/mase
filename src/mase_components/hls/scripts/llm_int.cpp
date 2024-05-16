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
typedef ap_int<8> ITYPE;
#define X_ROW 1
#define X_COL 4096
#define W_ROW X_COL
#define W_COL 11008
// High-precision cols
#define HP_COL 353
#define HP_DIM (HP_COL + 15) / 16 * 16
#define LP_DIM (X_COL - HP_COL + 15) / 16 * 16
#define THRESHOLD 0.001
#define CX 0.001
#define CW 0.001

void hp_mm(FTYPE xh[X_ROW][HP_DIM], FTYPE wh[HP_DIM][W_COL],
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
      for (int j = 0; j < HP_DIM; j += 16) {
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

void lp_mm(FTYPE xl[X_ROW][LP_DIM], FTYPE wl[LP_DIM][W_COL],
           ITYPE iy[X_ROW][W_COL]) {
#pragma HLS array_partition variable = xl type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wl type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = iy type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF

  for (int i = 0; i < X_ROW; i++) {
#pragma HLS LOOP_FLATTEN
    for (int k = 0; k < W_COL; k++) {
#pragma HLS LOOP_FLATTEN
      ITYPE res = 0;
      for (int j = 0; j < LP_DIM; j += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        ITYPE x_temp_0 = xl[i][j] * 127 / CX;
        ITYPE x_temp_1 = xl[i][j + 1] * 127 / CX;
        ITYPE x_temp_2 = xl[i][j + 2] * 127 / CX;
        ITYPE x_temp_3 = xl[i][j + 3] * 127 / CX;
        ITYPE x_temp_4 = xl[i][j + 4] * 127 / CX;
        ITYPE x_temp_5 = xl[i][j + 5] * 127 / CX;
        ITYPE x_temp_6 = xl[i][j + 6] * 127 / CX;
        ITYPE x_temp_7 = xl[i][j + 7] * 127 / CX;
        ITYPE x_temp_8 = xl[i][j + 8] * 127 / CX;
        ITYPE x_temp_9 = xl[i][j + 9] * 127 / CX;
        ITYPE x_temp_10 = xl[i][j + 10] * 127 / CX;
        ITYPE x_temp_11 = xl[i][j + 11] * 127 / CX;
        ITYPE x_temp_12 = xl[i][j + 12] * 127 / CX;
        ITYPE x_temp_13 = xl[i][j + 13] * 127 / CX;
        ITYPE x_temp_14 = xl[i][j + 14] * 127 / CX;
        ITYPE x_temp_15 = xl[i][j + 15] * 127 / CX;
        ITYPE w_temp_0 = wl[j][k] * 127 / CX;
        ITYPE w_temp_1 = wl[j + 1][k] * 127 / CX;
        ITYPE w_temp_2 = wl[j + 2][k] * 127 / CX;
        ITYPE w_temp_3 = wl[j + 3][k] * 127 / CX;
        ITYPE w_temp_4 = wl[j + 4][k] * 127 / CX;
        ITYPE w_temp_5 = wl[j + 5][k] * 127 / CX;
        ITYPE w_temp_6 = wl[j + 6][k] * 127 / CX;
        ITYPE w_temp_7 = wl[j + 7][k] * 127 / CX;
        ITYPE w_temp_8 = wl[j + 8][k] * 127 / CX;
        ITYPE w_temp_9 = wl[j + 9][k] * 127 / CX;
        ITYPE w_temp_10 = wl[j + 10][k] * 127 / CX;
        ITYPE w_temp_11 = wl[j + 11][k] * 127 / CX;
        ITYPE w_temp_12 = wl[j + 12][k] * 127 / CX;
        ITYPE w_temp_13 = wl[j + 13][k] * 127 / CX;
        ITYPE w_temp_14 = wl[j + 14][k] * 127 / CX;
        ITYPE w_temp_15 = wl[j + 15][k] * 127 / CX;
#pragma HLS bind_op variable = x_temp_0 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_1 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_2 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_3 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_4 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_5 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_6 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_7 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_8 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_9 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_10 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_11 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_12 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_13 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_14 op = hmul impl = fabric
#pragma HLS bind_op variable = x_temp_15 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_0 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_1 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_2 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_3 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_4 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_5 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_6 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_7 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_8 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_9 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_10 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_11 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_12 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_13 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_14 op = hmul impl = fabric
#pragma HLS bind_op variable = w_temp_15 op = hmul impl = fabric
        res += x_temp_0 * w_temp_0 + x_temp_1 * w_temp_1 + x_temp_2 * w_temp_2 +
               x_temp_3 * w_temp_3 + x_temp_4 * w_temp_4 + x_temp_5 * w_temp_5 +
               x_temp_6 * w_temp_6 + x_temp_7 * w_temp_7 + x_temp_8 * w_temp_8 +
               x_temp_9 * w_temp_9 + x_temp_10 * w_temp_10 +
               x_temp_11 * w_temp_11 + x_temp_12 * w_temp_12 +
               x_temp_13 * w_temp_13 + x_temp_14 * w_temp_14 +
               x_temp_15 * w_temp_15;
#pragma HLS bind_op variable = res op = mul impl = fabric
#pragma HLS bind_op variable = res op = add impl = fabric
      }
      iy[i][k] = res;
    }
  }
}

void gather(FTYPE y[X_ROW][W_COL], ITYPE iy[X_ROW][W_COL]) {
#pragma HLS array_partition variable = y type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = iy type = cyclic factor = 16 dim = 2
#pragma HLS INLINE OFF
  for (int i = 0; i < X_ROW; i++) {
#pragma HLS LOOP_FLATTEN
    for (int k = 0; k < W_COL; k += 16) {
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
      auto temp_0 = y[i][k];
      auto temp_1 = y[i][k + 1];
      auto temp_2 = y[i][k + 2];
      auto temp_3 = y[i][k + 3];
      auto temp_4 = y[i][k + 4];
      auto temp_5 = y[i][k + 5];
      auto temp_6 = y[i][k + 6];
      auto temp_7 = y[i][k + 7];
      auto temp_8 = y[i][k + 8];
      auto temp_9 = y[i][k + 9];
      auto temp_10 = y[i][k + 10];
      auto temp_11 = y[i][k + 11];
      auto temp_12 = y[i][k + 12];
      auto temp_13 = y[i][k + 13];
      auto temp_14 = y[i][k + 14];
      auto temp_15 = y[i][k + 15];
      temp_0 += (half)iy[i][k] * (half)CX * (half)CW / (half)(127 * 127);
      temp_1 += (half)iy[i][k + 1] * (half)CX * (half)CW / (half)(127 * 127);
      temp_2 += (half)iy[i][k + 2] * (half)CX * (half)CW / (half)(127 * 127);
      temp_3 += (half)iy[i][k + 3] * (half)CX * (half)CW / (half)(127 * 127);
      temp_4 += (half)iy[i][k + 4] * (half)CX * (half)CW / (half)(127 * 127);
      temp_5 += (half)iy[i][k + 5] * (half)CX * (half)CW / (half)(127 * 127);
      temp_6 += (half)iy[i][k + 6] * (half)CX * (half)CW / (half)(127 * 127);
      temp_7 += (half)iy[i][k + 7] * (half)CX * (half)CW / (half)(127 * 127);
      temp_8 += (half)iy[i][k + 8] * (half)CX * (half)CW / (half)(127 * 127);
      temp_9 += (half)iy[i][k + 9] * (half)CX * (half)CW / (half)(127 * 127);
      temp_10 += (half)iy[i][k + 10] * (half)CX * (half)CW / (half)(127 * 127);
      temp_11 += (half)iy[i][k + 11] * (half)CX * (half)CW / (half)(127 * 127);
      temp_12 += (half)iy[i][k + 12] * (half)CX * (half)CW / (half)(127 * 127);
      temp_13 += (half)iy[i][k + 13] * (half)CX * (half)CW / (half)(127 * 127);
      temp_14 += (half)iy[i][k + 14] * (half)CX * (half)CW / (half)(127 * 127);
      temp_15 += (half)iy[i][k + 15] * (half)CX * (half)CW / (half)(127 * 127);
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
#pragma HLS bind_op variable = temp_0 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_1 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_2 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_3 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_4 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_5 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_6 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_7 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_8 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_9 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_10 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_11 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_12 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_13 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_14 op = hadd impl = fabric
#pragma HLS bind_op variable = temp_15 op = hadd impl = fabric
      y[i][k] = temp_0;
      y[i][k + 1] = temp_1;
      y[i][k + 2] = temp_2;
      y[i][k + 3] = temp_3;
      y[i][k + 4] = temp_4;
      y[i][k + 5] = temp_5;
      y[i][k + 6] = temp_6;
      y[i][k + 7] = temp_7;
      y[i][k + 8] = temp_8;
      y[i][k + 9] = temp_9;
      y[i][k + 10] = temp_10;
      y[i][k + 11] = temp_11;
      y[i][k + 12] = temp_12;
      y[i][k + 13] = temp_13;
      y[i][k + 14] = temp_14;
      y[i][k + 15] = temp_15;
    }
  }
}

void scatter(FTYPE x[X_ROW][X_COL], FTYPE w[W_ROW][W_COL],
             FTYPE xh[X_ROW][HP_DIM], FTYPE wh[HP_DIM][W_COL],
             FTYPE xl[X_ROW][LP_DIM], FTYPE wl[LP_DIM][W_COL]) {
#pragma HLS array_partition variable = xl type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wl type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = xh type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wh type = cyclic factor = 16 dim = 1
#pragma HLS INLINE OFF

  // Assume the number of high-precision columns is HP_COL
  int il = 0, ih = 0;
  for (int i = 0; i < X_COL; i++) {
    if (x[i][0] > THRESHOLD) {
      for (int j = 0; j < X_ROW; j++)
#pragma HLS PIPELINE II = 1
        xh[j][ih] = x[j][i];
      for (int j = 0; j < W_COL; j++)
#pragma HLS PIPELINE II = 1
        wh[ih][j] = w[i][j];
      ih++;
    } else {
      for (int j = 0; j < X_ROW; j++)
#pragma HLS PIPELINE II = 1
        xl[j][il] = x[j][i];
      for (int j = 0; j < W_COL; j++)
#pragma HLS PIPELINE II = 1
        wh[il][j] = w[i][j];
      il++;
    }
  }
}

void top(FTYPE x[X_ROW][X_COL], FTYPE w[W_ROW][W_COL], FTYPE y[X_ROW][W_COL]) {

  FTYPE xh[X_ROW][HP_DIM];
  FTYPE wh[HP_DIM][W_COL];
  FTYPE xl[X_ROW][LP_DIM];
  FTYPE wl[LP_DIM][W_COL];
  ITYPE iy[X_ROW][W_COL];
#pragma HLS array_partition variable = xl type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wl type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = xh type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = wh type = cyclic factor = 16 dim = 1
#pragma HLS array_partition variable = y type = cyclic factor = 16 dim = 2
#pragma HLS array_partition variable = iy type = cyclic factor = 16 dim = 2

  scatter(x, w, xh, wh, xl, wl);
  hp_mm(xh, wh, y);
  lp_mm(xl, wl, iy);
  gather(y, iy);
}
