import os, sys

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
    )
)

from hls.bfp_arith import bfp_mm_gen
from hls.bfp_arith import bfp_add_gen
from hls import HLSWriter


def get_big_little_bfp(
    HIGH_MAN_WIDTH=7,
    LOW_MAN_WIDTH=3,
    X_ROW=1,
    X_COL=4096,
    W_COL=11008,
    A_COL=32,
):
    W_ROW = X_COL
    A_ROW = X_COL
    B_ROW = A_COL
    B_COL = W_COL

    writer = HLSWriter()
    # Due to opposite implementation
    # X is on the left and W is on the right
    # The hardware is WX so we use opposite parameters here

    # mm1
    bfp_mm_gen(
        writer,
        # this is x
        w_exp_width=8,
        w_man_width=7,
        w_row=1,
        w_col=16,
        w_row_depth=X_ROW,
        w_col_depth=int(X_COL / 16),
        # this is y
        x_exp_width=8,
        x_man_width=3,
        x_row=16,
        x_col=1,
        x_row_depth=int(W_ROW / 16),
        x_col_depth=W_COL,
    )

    # mm2
    bfp_mm_gen(
        writer,
        # this is x
        w_exp_width=8,
        w_man_width=7,
        w_row=1,
        w_col=16,
        w_row_depth=X_ROW,
        w_col_depth=int(X_COL / 16),
        # this is y
        x_exp_width=8,
        x_man_width=7,
        x_row=16,
        x_col=1,
        x_row_depth=int(A_ROW / 16),
        x_col_depth=A_COL,
    )

    # mm3
    bfp_mm_gen(
        writer,
        # this is x
        w_exp_width=8,
        w_man_width=7,
        w_row=1,
        w_col=16,
        w_row_depth=X_ROW,
        w_col_depth=int(A_COL / 16),
        # this is y
        x_exp_width=8,
        x_man_width=7,
        x_row=16,
        x_col=1,
        x_row_depth=int(B_ROW / 16),
        x_col_depth=B_COL,
    )

    # add
    bfp_add_gen(
        writer,
        # this is x
        w_exp_width=8,
        w_man_width=7,
        # this is y
        x_exp_width=8,
        x_man_width=7,
        x_row=16,
        x_col=1,
        x_row_depth=X_ROW,
        x_col_depth=int(W_COL / 16),
    )
    file_path = "bl_bfp.h"
    writer.emit(file_path)
    os.system("clang-format -i {}".format(file_path))

    emit_top_level(
        HIGH_MAN_WIDTH=HIGH_MAN_WIDTH,
        LOW_MAN_WIDTH=LOW_MAN_WIDTH,
        X_ROW=X_ROW,
        X_COL=X_COL,
        W_COL=W_COL,
        A_COL=A_COL,
        W_ROW=X_COL,
        A_ROW=X_COL,
        B_ROW=A_COL,
        B_COL=W_COL,
    )


def emit_top_level(
    HIGH_MAN_WIDTH=-1,
    LOW_MAN_WIDTH=-1,
    X_ROW=-1,
    X_COL=-1,
    W_COL=-1,
    A_COL=-1,
    W_ROW=-1,
    A_ROW=-1,
    B_ROW=-1,
    B_COL=-1,
):
    top = """
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

#include "bl_bfp.h"

void cast_xa(
  hls::stream<fixed_1_1_8_7_t> &xa,
  hls::stream<fixed_1_16_8_7_t> &xa_c) {
  for (int j = 0; j < 100; j++) {
    fixed_1_1_8_7_t x[16];
    fixed_1_16_8_7_t c;
    ap_uint<8> exponent = 0;
    for (int i = 0; i < 16; i++) {
      x[i] = xa.read();
      exponent = (exponent < x[i].exponent) ? x[i].exponent : exponent;
    }
    c.exponent = exponent;
    ap_int<8> man[16];
    for (int i = 0; i < 16; i++) {
      man[i] = x[i].data_0_0 >> (exponent - x[i].exponent);
    }
    c.data_0_0 = man [0 ];
    c.data_0_1 = man [1 ];
    c.data_0_2 = man [2 ];
    c.data_0_3 = man [3 ];
    c.data_0_4 = man [4 ];
    c.data_0_5 = man [5 ];
    c.data_0_6 = man [6 ];
    c.data_0_7 = man [7 ];
    c.data_0_8 = man [8 ];
    c.data_0_9 = man [9 ];
    c.data_0_10 = man[10];
    c.data_0_11 = man[11];
    c.data_0_12 = man[12];
    c.data_0_13 = man[13];
    c.data_0_14 = man[14];
    c.data_0_15 = man[15];
    xa_c.write(c);
  }
}

void cast_x(
  hls::stream<fixed_1_1_8_7_t> &xa,
  hls::stream<fixed_16_1_8_7_t> &xa_c) {
  for (int j = 0; j < 100; j++) {
    fixed_1_1_8_7_t x[16];
    fixed_16_1_8_7_t c;
    ap_uint<8> exponent = 0;
    for (int i = 0; i < 16; i++) {
      x[i] = xa.read();
      exponent = (exponent < x[i].exponent) ? x[i].exponent : exponent;
    }
    c.exponent = exponent;
    ap_int<8> man[16];
    for (int i = 0; i < 16; i++) {
      man[i] = x[i].data_0_0 >> (exponent - x[i].exponent);
    }
    c.data_0_0= man [0 ];
    c.data_1_0= man [1 ];
    c.data_2_0= man [2 ];
    c.data_3_0= man [3 ];
    c.data_4_0= man [4 ];
    c.data_5_0= man [5 ];
    c.data_6_0= man [6 ];
    c.data_7_0= man [7 ];
    c.data_8_0= man [8 ];
    c.data_9_0= man [9 ];
    c.data_10_0 = man[10];
    c.data_11_0 = man[11];
    c.data_12_0 = man[12];
    c.data_13_0 = man[13];
    c.data_14_0 = man[14];
    c.data_15_0 = man[15];
    xa_c.write(c);
  }
}

void top(
hls::stream<fixed_16_1_8_3_t> &w,
hls::stream<fixed_1_16_8_7_t> &x,
hls::stream<fixed_16_1_8_7_t> &a,
hls::stream<fixed_16_1_8_7_t> &b,
hls::stream<fixed_16_1_8_7_t> &y
) {

  hls::stream<fixed_1_1_8_7_t> x1;
  hls::stream<fixed_16_1_8_7_t> x1_c;
  hls::stream<fixed_1_1_8_7_t> xa;
  hls::stream<fixed_1_16_8_7_t> xa_c;
  hls::stream<fixed_1_1_8_7_t> x2;
  hls::stream<fixed_16_1_8_7_t> x2_c;

  bfp_mm_0(w, x, x1);
  bfp_mm_1(a, x, xa);
  cast_xa(xa, xa_c);
  bfp_mm_2(b, xa_c, x2);
  cast_x(x1, x1_c);
  cast_x(x2, x2_c);
  bfp_add_5(x1_c, x2_c, y);
}
"""
    file_path = "bl_bfp.cpp"
    f = open(file_path, "w")
    f.write(top)
    f.close()
    os.system("clang-format -i {}".format(file_path))


if __name__ == "__main__":
    get_big_little_bfp()
