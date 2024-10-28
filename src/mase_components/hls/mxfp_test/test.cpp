
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
struct fixed_16_1_8_3_t {
  ap_uint<8> exponent;
  ap_int<4> data_0_0;
  ap_int<4> data_1_0;
  ap_int<4> data_2_0;
  ap_int<4> data_3_0;
  ap_int<4> data_4_0;
  ap_int<4> data_5_0;
  ap_int<4> data_6_0;
  ap_int<4> data_7_0;
  ap_int<4> data_8_0;
  ap_int<4> data_9_0;
  ap_int<4> data_10_0;
  ap_int<4> data_11_0;
  ap_int<4> data_12_0;
  ap_int<4> data_13_0;
  ap_int<4> data_14_0;
  ap_int<4> data_15_0;
};
struct fixed_1_16_8_3_t {
  ap_uint<8> exponent;
  ap_int<4> data_0_0;
  ap_int<4> data_0_1;
  ap_int<4> data_0_2;
  ap_int<4> data_0_3;
  ap_int<4> data_0_4;
  ap_int<4> data_0_5;
  ap_int<4> data_0_6;
  ap_int<4> data_0_7;
  ap_int<4> data_0_8;
  ap_int<4> data_0_9;
  ap_int<4> data_0_10;
  ap_int<4> data_0_11;
  ap_int<4> data_0_12;
  ap_int<4> data_0_13;
  ap_int<4> data_0_14;
  ap_int<4> data_0_15;
};
struct fixed_2_1_9_7_t {
  ap_uint<9> exponent;
  ap_int<8> data_0_0;
  ap_int<8> data_1_0;
};

// E4M3
ap_int<8> fp8_add(ap_int<8> x, ap_int<8> y) {
#pragma HLS INLINE

  union {
    int intval;
    half fpval;
  } xin;

  union {
    int intval;
    half fpval;
  } yin;

  union {
    int intval;
    half fpval;
  } zout;

  xin.intval = (x.range(7, 3), (ap_int<1>)(0), x.range(2, 0), (ap_int<7>)(0));
  yin.intval = (x.range(7, 3), (ap_int<1>)(0), x.range(2, 0), (ap_int<7>)(0));

  zout.fpval = xin.fpval + yin.fpval;

  ap_int<16> z = zout.intval;
  return (z[15], z.range(13, 7));

  // ap_uint<4> exp_x = x.range(6, 3);
  // ap_uint<4> exp_y = y.range(6, 3);
  // ap_uint<3> man_x = x.range(2, 0);
  // ap_uint<3> man_y = y.range(2, 0);

  // ap_int<4> man_xs = (x[7], man_x);
  // ap_int<4> man_ys = (y[7], man_y);

  // ap_int<5> man_r;
  // ap_int<5> exp_r;

  // if (exp_x > exp_y) {
  //   ap_int<4> shift = exp_x - exp_y;
  //   if (shift == 1)
  //     man_y >>= 1;
  //   else if (shift == 2)
  //     man_y >>= 2;
  //   else
  //     man_y = 0;

  // } else {
  //   ap_int<4> shift = exp_y - exp_x;
  //   if (shift == 1)
  //     man_x >>= 1;
  //   else if (shift == 2)
  //     man_x >>= 2;
  //   else
  //     man_x = 0;
  // }

  // man_r = man_x + man_y;
  // if (man_r[3] == 1) {
  //   exp_r = exp_x + 1;
  //   man_r >>= 1;
  // }

  // // saturation
  // if (exp_r[4] == 1)
  //   exp_r = 0xf;

  // return (man_r[4] && y[7], exp_r.range(3, 0), man_r.range(2, 0));
}

// E4M3
ap_int<8> fp8_mult(ap_int<8> x, ap_int<8> y) {
#pragma HLS INLINE

  union {
    int intval;
    half fpval;
  } xin;

  union {
    int intval;
    half fpval;
  } yin;

  union {
    int intval;
    half fpval;
  } zout;

  xin.intval = (x.range(7, 3), (ap_int<1>)(0), x.range(2, 0), (ap_int<7>)(0));
  yin.intval = (x.range(7, 3), (ap_int<1>)(0), x.range(2, 0), (ap_int<7>)(0));

  zout.fpval = xin.fpval * yin.fpval;

  ap_int<16> z = zout.intval;
  return (z[15], z.range(13, 7));

  // ap_int<4> exp_x = x.range(6, 3);
  // ap_int<4> exp_y = y.range(6, 3);
  // ap_int<3> man_x = x.range(2, 0);
  // ap_int<3> man_y = y.range(2, 0);

  // ap_int<6> man_r = man_x * man_y;
  // ap_int<5> exp_r = exp_x + exp_y;

  // if (man_r[5])
  //   exp_r += 3;
  // else if (man_r[4])
  //   exp_r += 2;
  // else if (man_r[3])
  //   exp_r += 1;

  // // saturation
  // if (exp_r[4] == 1)
  //   exp_r = 0xf;

  // return (x[7] && y[7], exp_r.range(3, 0), man_r.range(2, 0));
}

// Linear 2D:
void mxfp_linear2d_0(hls::stream<fixed_16_1_8_3_t> &data_in,
                     hls::stream<fixed_2_1_9_7_t> &data_out) {
#pragma HLS INLINE OFF
  fixed_1_16_8_3_t weight_0[8][8];
  fixed_1_16_8_3_t weight_1[8][8];
  fixed_16_1_8_3_t d;
  fixed_2_1_9_7_t dout_buff[8];

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        if (k == 0) {
          d = data_in.read();
          if (j != 7) {
            ap_uint<9> wx_exp_0_0 = d.exponent + weight_0[j][k].exponent;
            ap_uint<10> wx_man_0_0 = fp8_add(
                fp8_mult(weight_0[j][k].data_0_0, d.data_0_0),
                fp8_add(
                    fp8_mult(weight_0[j][k].data_0_1, d.data_1_0),
                    fp8_add(
                        fp8_mult(weight_0[j][k].data_0_2, d.data_2_0),
                        fp8_add(
                            fp8_mult(weight_0[j][k].data_0_3, d.data_3_0),
                            fp8_add(
                                fp8_mult(weight_0[j][k].data_0_4, d.data_4_0),
                                fp8_add(
                                    fp8_mult(weight_0[j][k].data_0_5,
                                             d.data_5_0),
                                    fp8_add(
                                        fp8_mult(weight_0[j][k].data_0_6,
                                                 d.data_6_0),
                                        fp8_add(
                                            fp8_mult(weight_0[j][k].data_0_7,
                                                     d.data_7_0),
                                            fp8_add(
                                                fp8_mult(
                                                    weight_0[j][k].data_0_8,
                                                    d.data_8_0),
                                                fp8_add(
                                                    fp8_mult(
                                                        weight_0[j][k].data_0_9,
                                                        d.data_9_0),
                                                    fp8_add(
                                                        fp8_mult(weight_0[j][k]
                                                                     .data_0_10,
                                                                 d.data_10_0),
                                                        fp8_add(
                                                            fp8_mult(
                                                                weight_0[j][k]
                                                                    .data_0_11,
                                                                d.data_11_0),
                                                            fp8_add(
                                                                fp8_mult(
                                                                    weight_0[j][k]
                                                                        .data_0_12,
                                                                    d.data_12_0),
                                                                fp8_add(
                                                                    fp8_mult(
                                                                        weight_0
                                                                            [j]
                                                                            [k]
                                                                                .data_0_13,
                                                                        d.data_13_0),
                                                                    fp8_add(
                                                                        fp8_mult(
                                                                            weight_0
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_14,
                                                                            d.data_14_0),
                                                                        fp8_mult(
                                                                            weight_0
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_15,
                                                                            d.data_15_0))))))))))))))));
            if (wx_man_0_0[9])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[8])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[7])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[6])
              wx_exp_0_0 += 1;
            ap_uint<7> wx_cast_man_0_0 = wx_man_0_0.range(5, -1);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout_buff[k].data_0_0 += wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 3) {
                ap_uint<9> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 2);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 3) {
                ap_uint<9> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 1) + wx_cast_man_0_0;
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 2) + wx_cast_man_0_0;
                }
              }
            }
            ap_uint<9> wx_exp_1_0 = d.exponent + weight_1[j][k].exponent;
            ap_uint<10> wx_man_1_0 = fp8_add(
                fp8_mult(weight_1[j][k].data_0_0, d.data_0_0),
                fp8_add(
                    fp8_mult(weight_1[j][k].data_0_1, d.data_1_0),
                    fp8_add(
                        fp8_mult(weight_1[j][k].data_0_2, d.data_2_0),
                        fp8_add(
                            fp8_mult(weight_1[j][k].data_0_3, d.data_3_0),
                            fp8_add(
                                fp8_mult(weight_1[j][k].data_0_4, d.data_4_0),
                                fp8_add(
                                    fp8_mult(weight_1[j][k].data_0_5,
                                             d.data_5_0),
                                    fp8_add(
                                        fp8_mult(weight_1[j][k].data_0_6,
                                                 d.data_6_0),
                                        fp8_add(
                                            fp8_mult(weight_1[j][k].data_0_7,
                                                     d.data_7_0),
                                            fp8_add(
                                                fp8_mult(
                                                    weight_1[j][k].data_0_8,
                                                    d.data_8_0),
                                                fp8_add(
                                                    fp8_mult(
                                                        weight_1[j][k].data_0_9,
                                                        d.data_9_0),
                                                    fp8_add(
                                                        fp8_mult(weight_1[j][k]
                                                                     .data_0_10,
                                                                 d.data_10_0),
                                                        fp8_add(
                                                            fp8_mult(
                                                                weight_1[j][k]
                                                                    .data_0_11,
                                                                d.data_11_0),
                                                            fp8_add(
                                                                fp8_mult(
                                                                    weight_1[j][k]
                                                                        .data_0_12,
                                                                    d.data_12_0),
                                                                fp8_add(
                                                                    fp8_mult(
                                                                        weight_1
                                                                            [j]
                                                                            [k]
                                                                                .data_0_13,
                                                                        d.data_13_0),
                                                                    fp8_add(
                                                                        fp8_mult(
                                                                            weight_1
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_14,
                                                                            d.data_14_0),
                                                                        fp8_mult(
                                                                            weight_1
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_15,
                                                                            d.data_15_0))))))))))))))));
            if (wx_man_1_0[9])
              wx_exp_1_0 += 4;
            else if (wx_man_1_0[8])
              wx_exp_1_0 += 3;
            else if (wx_man_1_0[7])
              wx_exp_1_0 += 2;
            else if (wx_man_1_0[6])
              wx_exp_1_0 += 1;
            ap_uint<7> wx_cast_man_1_0 = wx_man_1_0.range(5, -1);
            {
              auto exp_0 = wx_exp_1_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout_buff[k].data_1_0 += wx_cast_man_1_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 3) {
                ap_uint<9> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout_buff[k].data_1_0 += (wx_cast_man_1_0 >> 1);
                }

                if (diff == 2) {
                  dout_buff[k].data_1_0 += (wx_cast_man_1_0 >> 2);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 3) {
                ap_uint<9> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout_buff[k].data_1_0 =
                      (dout_buff[k].data_1_0 >> 1) + wx_cast_man_1_0;
                }

                if (diff == 2) {
                  dout_buff[k].data_1_0 =
                      (dout_buff[k].data_1_0 >> 2) + wx_cast_man_1_0;
                }
              }
            }
          } else {
            fixed_2_1_9_7_t dout;
            ap_uint<9> wx_exp_0_0 = d.exponent + weight_0[j][k].exponent;
            ap_uint<10> wx_man_0_0 = fp8_add(
                fp8_mult(weight_0[j][k].data_0_0, d.data_0_0),
                fp8_add(
                    fp8_mult(weight_0[j][k].data_0_1, d.data_1_0),
                    fp8_add(
                        fp8_mult(weight_0[j][k].data_0_2, d.data_2_0),
                        fp8_add(
                            fp8_mult(weight_0[j][k].data_0_3, d.data_3_0),
                            fp8_add(
                                fp8_mult(weight_0[j][k].data_0_4, d.data_4_0),
                                fp8_add(
                                    fp8_mult(weight_0[j][k].data_0_5,
                                             d.data_5_0),
                                    fp8_add(
                                        fp8_mult(weight_0[j][k].data_0_6,
                                                 d.data_6_0),
                                        fp8_add(
                                            fp8_mult(weight_0[j][k].data_0_7,
                                                     d.data_7_0),
                                            fp8_add(
                                                fp8_mult(
                                                    weight_0[j][k].data_0_8,
                                                    d.data_8_0),
                                                fp8_add(
                                                    fp8_mult(
                                                        weight_0[j][k].data_0_9,
                                                        d.data_9_0),
                                                    fp8_add(
                                                        fp8_mult(weight_0[j][k]
                                                                     .data_0_10,
                                                                 d.data_10_0),
                                                        fp8_add(
                                                            fp8_mult(
                                                                weight_0[j][k]
                                                                    .data_0_11,
                                                                d.data_11_0),
                                                            fp8_add(
                                                                fp8_mult(
                                                                    weight_0[j][k]
                                                                        .data_0_12,
                                                                    d.data_12_0),
                                                                fp8_add(
                                                                    fp8_mult(
                                                                        weight_0
                                                                            [j]
                                                                            [k]
                                                                                .data_0_13,
                                                                        d.data_13_0),
                                                                    fp8_add(
                                                                        fp8_mult(
                                                                            weight_0
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_14,
                                                                            d.data_14_0),
                                                                        fp8_mult(
                                                                            weight_0
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_15,
                                                                            d.data_15_0))))))))))))))));
            if (wx_man_0_0[9])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[8])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[7])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[6])
              wx_exp_0_0 += 1;
            ap_uint<7> wx_cast_man_0_0 = wx_man_0_0.range(5, -1);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout.data_0_0 = dout_buff[k].data_0_0 + wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 3) {
                ap_uint<9> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 2);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 3) {
                ap_uint<9> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 1) + wx_cast_man_0_0;
                }

                if (diff == 2) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 2) + wx_cast_man_0_0;
                }
              }
            }
            dout_buff[k].exponent = 0;
            ap_uint<9> wx_exp_1_0 = d.exponent + weight_1[j][k].exponent;
            ap_uint<10> wx_man_1_0 = fp8_add(
                fp8_mult(weight_1[j][k].data_0_0, d.data_0_0),
                fp8_add(
                    fp8_mult(weight_1[j][k].data_0_1, d.data_1_0),
                    fp8_add(
                        fp8_mult(weight_1[j][k].data_0_2, d.data_2_0),
                        fp8_add(
                            fp8_mult(weight_1[j][k].data_0_3, d.data_3_0),
                            fp8_add(
                                fp8_mult(weight_1[j][k].data_0_4, d.data_4_0),
                                fp8_add(
                                    fp8_mult(weight_1[j][k].data_0_5,
                                             d.data_5_0),
                                    fp8_add(
                                        fp8_mult(weight_1[j][k].data_0_6,
                                                 d.data_6_0),
                                        fp8_add(
                                            fp8_mult(weight_1[j][k].data_0_7,
                                                     d.data_7_0),
                                            fp8_add(
                                                fp8_mult(
                                                    weight_1[j][k].data_0_8,
                                                    d.data_8_0),
                                                fp8_add(
                                                    fp8_mult(
                                                        weight_1[j][k].data_0_9,
                                                        d.data_9_0),
                                                    fp8_add(
                                                        fp8_mult(weight_1[j][k]
                                                                     .data_0_10,
                                                                 d.data_10_0),
                                                        fp8_add(
                                                            fp8_mult(
                                                                weight_1[j][k]
                                                                    .data_0_11,
                                                                d.data_11_0),
                                                            fp8_add(
                                                                fp8_mult(
                                                                    weight_1[j][k]
                                                                        .data_0_12,
                                                                    d.data_12_0),
                                                                fp8_add(
                                                                    fp8_mult(
                                                                        weight_1
                                                                            [j]
                                                                            [k]
                                                                                .data_0_13,
                                                                        d.data_13_0),
                                                                    fp8_add(
                                                                        fp8_mult(
                                                                            weight_1
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_14,
                                                                            d.data_14_0),
                                                                        fp8_mult(
                                                                            weight_1
                                                                                [j]
                                                                                [k]
                                                                                    .data_0_15,
                                                                            d.data_15_0))))))))))))))));
            if (wx_man_1_0[9])
              wx_exp_1_0 += 4;
            else if (wx_man_1_0[8])
              wx_exp_1_0 += 3;
            else if (wx_man_1_0[7])
              wx_exp_1_0 += 2;
            else if (wx_man_1_0[6])
              wx_exp_1_0 += 1;
            ap_uint<7> wx_cast_man_1_0 = wx_man_1_0.range(5, -1);
            {
              auto exp_0 = wx_exp_1_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout.data_1_0 = dout_buff[k].data_1_0 + wx_cast_man_1_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 3) {
                ap_uint<9> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout.data_1_0 +=
                      dout_buff[k].data_1_0 + (wx_cast_man_1_0 >> 1);
                }

                if (diff == 2) {
                  dout.data_1_0 +=
                      dout_buff[k].data_1_0 + (wx_cast_man_1_0 >> 2);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 3) {
                ap_uint<9> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout.data_1_0 =
                      (dout_buff[k].data_1_0 >> 1) + wx_cast_man_1_0;
                }

                if (diff == 2) {
                  dout.data_1_0 =
                      (dout_buff[k].data_1_0 >> 2) + wx_cast_man_1_0;
                }
              }
            }
            dout_buff[k].exponent = 0;
            dout_buff[k].data_0_0 = 0;
            dout_buff[k].data_1_0 = 0;
            data_out.write(dout);
          }
        }
      }
    }
  }
}
