
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
struct fixed_1_16_8_5_t {
  ap_uint<8> exponent;
  ap_int<6> data_0_0;
  ap_int<6> data_0_1;
  ap_int<6> data_0_2;
  ap_int<6> data_0_3;
  ap_int<6> data_0_4;
  ap_int<6> data_0_5;
  ap_int<6> data_0_6;
  ap_int<6> data_0_7;
  ap_int<6> data_0_8;
  ap_int<6> data_0_9;
  ap_int<6> data_0_10;
  ap_int<6> data_0_11;
  ap_int<6> data_0_12;
  ap_int<6> data_0_13;
  ap_int<6> data_0_14;
  ap_int<6> data_0_15;
};
struct fixed_1_1_8_5_t {
  ap_uint<8> exponent;
  ap_int<6> data_0_0;
};
struct fixed_16_1_8_5_t {
  ap_uint<8> exponent;
  ap_int<6> data_0_0;
  ap_int<6> data_1_0;
  ap_int<6> data_2_0;
  ap_int<6> data_3_0;
  ap_int<6> data_4_0;
  ap_int<6> data_5_0;
  ap_int<6> data_6_0;
  ap_int<6> data_7_0;
  ap_int<6> data_8_0;
  ap_int<6> data_9_0;
  ap_int<6> data_10_0;
  ap_int<6> data_11_0;
  ap_int<6> data_12_0;
  ap_int<6> data_13_0;
  ap_int<6> data_14_0;
  ap_int<6> data_15_0;
};
// Linear 2D:
void bfp_mm_0(hls::stream<fixed_16_1_8_3_t> &data_in,
              hls::stream<fixed_1_16_8_5_t> &weights,
              hls::stream<fixed_1_1_8_5_t> &data_out) {
#pragma HLS INLINE OFF
  fixed_16_1_8_3_t d;
  fixed_1_1_8_5_t dout_buff[1];

  for (int i = 0; i < 11008; i++) {
    for (int j = 0; j < 256; j++) {
      for (int k = 0; k < 1; k++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        auto weight = weights.read();
        if (k == 0) {
          d = data_in.read();
          if (j != 255) {
            ap_uint<9> wx_exp_0_0 = d.exponent + weight.exponent;
            ap_uint<12> wx_man_0_0 =
                weight.data_0_0 * d.data_0_0 + weight.data_0_1 * d.data_1_0 +
                weight.data_0_2 * d.data_2_0 + weight.data_0_3 * d.data_3_0 +
                weight.data_0_4 * d.data_4_0 + weight.data_0_5 * d.data_5_0 +
                weight.data_0_6 * d.data_6_0 + weight.data_0_7 * d.data_7_0 +
                weight.data_0_8 * d.data_8_0 + weight.data_0_9 * d.data_9_0 +
                weight.data_0_10 * d.data_10_0 +
                weight.data_0_11 * d.data_11_0 +
                weight.data_0_12 * d.data_12_0 +
                weight.data_0_13 * d.data_13_0 +
                weight.data_0_14 * d.data_14_0 + weight.data_0_15 * d.data_15_0;
            if (wx_man_0_0[11])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[10])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[9])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[8])
              wx_exp_0_0 += 1;
            ap_uint<5> wx_cast_man_0_0 = wx_man_0_0.range(7, 3);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout_buff[k].data_0_0 += wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 3) {
                ap_uint<8> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 2);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 3) {
                ap_uint<8> diff = exp_1 - exp_0;

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
          } else {
            fixed_1_1_8_5_t dout;
            ap_uint<9> wx_exp_0_0 = d.exponent + weight.exponent;
            ap_uint<12> wx_man_0_0 =
                weight.data_0_0 * d.data_0_0 + weight.data_0_1 * d.data_1_0 +
                weight.data_0_2 * d.data_2_0 + weight.data_0_3 * d.data_3_0 +
                weight.data_0_4 * d.data_4_0 + weight.data_0_5 * d.data_5_0 +
                weight.data_0_6 * d.data_6_0 + weight.data_0_7 * d.data_7_0 +
                weight.data_0_8 * d.data_8_0 + weight.data_0_9 * d.data_9_0 +
                weight.data_0_10 * d.data_10_0 +
                weight.data_0_11 * d.data_11_0 +
                weight.data_0_12 * d.data_12_0 +
                weight.data_0_13 * d.data_13_0 +
                weight.data_0_14 * d.data_14_0 + weight.data_0_15 * d.data_15_0;
            if (wx_man_0_0[11])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[10])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[9])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[8])
              wx_exp_0_0 += 1;
            ap_uint<5> wx_cast_man_0_0 = wx_man_0_0.range(7, 3);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout.data_0_0 = dout_buff[k].data_0_0 + wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 3) {
                ap_uint<8> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 2);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 3) {
                ap_uint<8> diff = exp_1 - exp_0;

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
            dout_buff[k].data_0_0 = 0;
            data_out.write(dout);
          }
        }
      }
    }
  }
}

// Linear 2D:
void bfp_mm_1(hls::stream<fixed_16_1_8_5_t> &data_in,
              hls::stream<fixed_1_16_8_5_t> &weights,
              hls::stream<fixed_1_1_8_5_t> &data_out) {
#pragma HLS INLINE OFF
  fixed_16_1_8_5_t d;
  fixed_1_1_8_5_t dout_buff[1];

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 256; j++) {
      for (int k = 0; k < 1; k++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        auto weight = weights.read();
        if (k == 0) {
          d = data_in.read();
          if (j != 255) {
            ap_uint<9> wx_exp_0_0 = d.exponent + weight.exponent;
            ap_uint<14> wx_man_0_0 =
                weight.data_0_0 * d.data_0_0 + weight.data_0_1 * d.data_1_0 +
                weight.data_0_2 * d.data_2_0 + weight.data_0_3 * d.data_3_0 +
                weight.data_0_4 * d.data_4_0 + weight.data_0_5 * d.data_5_0 +
                weight.data_0_6 * d.data_6_0 + weight.data_0_7 * d.data_7_0 +
                weight.data_0_8 * d.data_8_0 + weight.data_0_9 * d.data_9_0 +
                weight.data_0_10 * d.data_10_0 +
                weight.data_0_11 * d.data_11_0 +
                weight.data_0_12 * d.data_12_0 +
                weight.data_0_13 * d.data_13_0 +
                weight.data_0_14 * d.data_14_0 + weight.data_0_15 * d.data_15_0;
            if (wx_man_0_0[13])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[12])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[11])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[10])
              wx_exp_0_0 += 1;
            ap_uint<5> wx_cast_man_0_0 = wx_man_0_0.range(9, 5);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout_buff[k].data_0_0 += wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 5) {
                ap_uint<8> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 2);
                }

                if (diff == 3) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 3);
                }

                if (diff == 4) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 4);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 5) {
                ap_uint<8> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 1) + wx_cast_man_0_0;
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 2) + wx_cast_man_0_0;
                }

                if (diff == 3) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 3) + wx_cast_man_0_0;
                }

                if (diff == 4) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 4) + wx_cast_man_0_0;
                }
              }
            }
          } else {
            fixed_1_1_8_5_t dout;
            ap_uint<9> wx_exp_0_0 = d.exponent + weight.exponent;
            ap_uint<14> wx_man_0_0 =
                weight.data_0_0 * d.data_0_0 + weight.data_0_1 * d.data_1_0 +
                weight.data_0_2 * d.data_2_0 + weight.data_0_3 * d.data_3_0 +
                weight.data_0_4 * d.data_4_0 + weight.data_0_5 * d.data_5_0 +
                weight.data_0_6 * d.data_6_0 + weight.data_0_7 * d.data_7_0 +
                weight.data_0_8 * d.data_8_0 + weight.data_0_9 * d.data_9_0 +
                weight.data_0_10 * d.data_10_0 +
                weight.data_0_11 * d.data_11_0 +
                weight.data_0_12 * d.data_12_0 +
                weight.data_0_13 * d.data_13_0 +
                weight.data_0_14 * d.data_14_0 + weight.data_0_15 * d.data_15_0;
            if (wx_man_0_0[13])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[12])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[11])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[10])
              wx_exp_0_0 += 1;
            ap_uint<5> wx_cast_man_0_0 = wx_man_0_0.range(9, 5);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout.data_0_0 = dout_buff[k].data_0_0 + wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 5) {
                ap_uint<8> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 2);
                }

                if (diff == 3) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 3);
                }

                if (diff == 4) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 4);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 5) {
                ap_uint<8> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 1) + wx_cast_man_0_0;
                }

                if (diff == 2) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 2) + wx_cast_man_0_0;
                }

                if (diff == 3) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 3) + wx_cast_man_0_0;
                }

                if (diff == 4) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 4) + wx_cast_man_0_0;
                }
              }
            }
            dout_buff[k].exponent = 0;
            dout_buff[k].data_0_0 = 0;
            data_out.write(dout);
          }
        }
      }
    }
  }
}

// Linear 2D:
void bfp_mm_2(hls::stream<fixed_16_1_8_5_t> &data_in,
              hls::stream<fixed_1_16_8_5_t> &weights,
              hls::stream<fixed_1_1_8_5_t> &data_out) {
#pragma HLS INLINE OFF
  fixed_16_1_8_5_t d;
  fixed_1_1_8_5_t dout_buff[1];

  for (int i = 0; i < 11008; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 1; k++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
        auto weight = weights.read();
        if (k == 0) {
          d = data_in.read();
          if (j != 1) {
            ap_uint<9> wx_exp_0_0 = d.exponent + weight.exponent;
            ap_uint<14> wx_man_0_0 =
                weight.data_0_0 * d.data_0_0 + weight.data_0_1 * d.data_1_0 +
                weight.data_0_2 * d.data_2_0 + weight.data_0_3 * d.data_3_0 +
                weight.data_0_4 * d.data_4_0 + weight.data_0_5 * d.data_5_0 +
                weight.data_0_6 * d.data_6_0 + weight.data_0_7 * d.data_7_0 +
                weight.data_0_8 * d.data_8_0 + weight.data_0_9 * d.data_9_0 +
                weight.data_0_10 * d.data_10_0 +
                weight.data_0_11 * d.data_11_0 +
                weight.data_0_12 * d.data_12_0 +
                weight.data_0_13 * d.data_13_0 +
                weight.data_0_14 * d.data_14_0 + weight.data_0_15 * d.data_15_0;
            if (wx_man_0_0[13])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[12])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[11])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[10])
              wx_exp_0_0 += 1;
            ap_uint<5> wx_cast_man_0_0 = wx_man_0_0.range(9, 5);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout_buff[k].data_0_0 += wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 5) {
                ap_uint<8> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 2);
                }

                if (diff == 3) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 3);
                }

                if (diff == 4) {
                  dout_buff[k].data_0_0 += (wx_cast_man_0_0 >> 4);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 5) {
                ap_uint<8> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 1) + wx_cast_man_0_0;
                }

                if (diff == 2) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 2) + wx_cast_man_0_0;
                }

                if (diff == 3) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 3) + wx_cast_man_0_0;
                }

                if (diff == 4) {
                  dout_buff[k].data_0_0 =
                      (dout_buff[k].data_0_0 >> 4) + wx_cast_man_0_0;
                }
              }
            }
          } else {
            fixed_1_1_8_5_t dout;
            ap_uint<9> wx_exp_0_0 = d.exponent + weight.exponent;
            ap_uint<14> wx_man_0_0 =
                weight.data_0_0 * d.data_0_0 + weight.data_0_1 * d.data_1_0 +
                weight.data_0_2 * d.data_2_0 + weight.data_0_3 * d.data_3_0 +
                weight.data_0_4 * d.data_4_0 + weight.data_0_5 * d.data_5_0 +
                weight.data_0_6 * d.data_6_0 + weight.data_0_7 * d.data_7_0 +
                weight.data_0_8 * d.data_8_0 + weight.data_0_9 * d.data_9_0 +
                weight.data_0_10 * d.data_10_0 +
                weight.data_0_11 * d.data_11_0 +
                weight.data_0_12 * d.data_12_0 +
                weight.data_0_13 * d.data_13_0 +
                weight.data_0_14 * d.data_14_0 + weight.data_0_15 * d.data_15_0;
            if (wx_man_0_0[13])
              wx_exp_0_0 += 4;
            else if (wx_man_0_0[12])
              wx_exp_0_0 += 3;
            else if (wx_man_0_0[11])
              wx_exp_0_0 += 2;
            else if (wx_man_0_0[10])
              wx_exp_0_0 += 1;
            ap_uint<5> wx_cast_man_0_0 = wx_man_0_0.range(9, 5);
            {
              auto exp_0 = wx_exp_0_0;
              auto exp_1 = dout_buff[k].exponent;

              if (exp_0 == exp_1) {
                dout.data_0_0 = dout_buff[k].data_0_0 + wx_cast_man_0_0;
              } else if (exp_0 > exp_1 && exp_0 - exp_1 < 5) {
                ap_uint<8> diff = exp_0 - exp_1;

                if (diff == 1) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 1);
                }

                if (diff == 2) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 2);
                }

                if (diff == 3) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 3);
                }

                if (diff == 4) {
                  dout.data_0_0 +=
                      dout_buff[k].data_0_0 + (wx_cast_man_0_0 >> 4);
                }

              } else if (exp_1 > exp_0 && exp_1 - exp_0 < 5) {
                ap_uint<8> diff = exp_1 - exp_0;

                if (diff == 1) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 1) + wx_cast_man_0_0;
                }

                if (diff == 2) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 2) + wx_cast_man_0_0;
                }

                if (diff == 3) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 3) + wx_cast_man_0_0;
                }

                if (diff == 4) {
                  dout.data_0_0 =
                      (dout_buff[k].data_0_0 >> 4) + wx_cast_man_0_0;
                }
              }
            }
            dout_buff[k].exponent = 0;
            dout_buff[k].data_0_0 = 0;
            data_out.write(dout);
          }
        }
      }
    }
  }
}

// BFP Single Adder:
void bfp_adder_3(ap_uint<1> sign_0, ap_uint<8> exp_0, ap_uint<5> man_0,
                 ap_uint<1> sign_1, ap_uint<8> exp_1, ap_uint<5> man_1,
                 ap_uint<1> *sign_2, ap_uint<8> *exp_2, ap_uint<5> *man_2) {
#pragma HLS INLINE

  ap_uint<8> exp_2_temp;
  ap_uint<1> sign_2_temp;
  ap_uint<1> carry;
  ap_uint<1> zero = 0;
  ap_uint<5> man_2_temp;
  if (exp_0 == exp_1) {
    exp_2_temp = exp_0;
    (sign_2_temp, man_2_temp) = (sign_0, man_0) + (sign_1, man_1);
  } else if (exp_0 - exp_1 >= 5) {
    exp_2_temp = exp_0;
    (sign_2_temp, man_2_temp) = (sign_0, man_0);
  } else if (exp_1 - exp_0 >= 5) {
    exp_2_temp = exp_1;
    (sign_2_temp, man_2_temp) = (sign_1, man_1);
  } else if (exp_0 > exp_1) {
    ap_uint<8> diff = exp_0 - exp_1;

    if (diff == 1) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, man_0) + (sign_1, zero, (man_1 >> 1));
      exp_2_temp = exp_0 + carry;
    }

    if (diff == 2) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, man_0) + (sign_1, zero, (man_1 >> 2));
      exp_2_temp = exp_0 + carry;
    }

    if (diff == 3) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, man_0) + (sign_1, zero, (man_1 >> 3));
      exp_2_temp = exp_0 + carry;
    }

    if (diff == 4) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, man_0) + (sign_1, zero, (man_1 >> 4));
      exp_2_temp = exp_0 + carry;
    }

  } else {
    ap_uint<8> diff = exp_1 - exp_0;

    if (diff == 1) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, (man_0 >> 1)) + (sign_1, zero, man_1);
      exp_2_temp = exp_1 + carry;
    }

    if (diff == 2) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, (man_0 >> 2)) + (sign_1, zero, man_1);
      exp_2_temp = exp_1 + carry;
    }

    if (diff == 3) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, (man_0 >> 3)) + (sign_1, zero, man_1);
      exp_2_temp = exp_1 + carry;
    }

    if (diff == 4) {
      (sign_2_temp, carry, man_2_temp) =
          (sign_0, zero, (man_0 >> 4)) + (sign_1, zero, man_1);
      exp_2_temp = exp_1 + carry;
    }
  }
  *exp_2 = exp_2_temp;
  *sign_2 = sign_2_temp;
  *man_2 = man_2_temp;
}

// BFP Add Block:
void bfp_block_adder_4(fixed_16_1_8_5_t d0, fixed_16_1_8_5_t d1,
                       fixed_16_1_8_5_t *data) {
#pragma HLS INLINE

  ap_uint<8> max_exp = 0;

  ap_uint<1> sign_0_0_0 = d0.data_0_0[5];
  ap_uint<8> exp_0_0_0 = d0.exponent;
  ap_uint<5> man_0_0_0 = d0.data_0_0.range(5, 0);
  ap_uint<1> sign_0_0_1 = d0.data_0_0[5];
  ap_uint<8> exp_0_0_1 = d0.exponent;
  ap_uint<5> man_0_0_1 = d0.data_0_0.range(5, 0);
  ap_uint<1> sign_0_0_2;
  ap_uint<8> exp_0_0_2;
  ap_uint<5> man_0_0_2;
  bfp_adder_3(sign_0_0_0, exp_0_0_0, man_0_0_0, sign_0_0_1, exp_0_0_1,
              man_0_0_1, &sign_0_0_2, &exp_0_0_2, &man_0_0_2);

  ap_uint<8> res_0_0_exp = exp_0_0_2;
  ap_int<6> res_0_0_man = (sign_0_0_2, man_0_0_2);
  max_exp = (max_exp > exp_0_0_2) ? max_exp : exp_0_0_2;

  ap_uint<1> sign_1_0_0 = d0.data_1_0[5];
  ap_uint<8> exp_1_0_0 = d0.exponent;
  ap_uint<5> man_1_0_0 = d0.data_1_0.range(5, 0);
  ap_uint<1> sign_1_0_1 = d0.data_1_0[5];
  ap_uint<8> exp_1_0_1 = d0.exponent;
  ap_uint<5> man_1_0_1 = d0.data_1_0.range(5, 0);
  ap_uint<1> sign_1_0_2;
  ap_uint<8> exp_1_0_2;
  ap_uint<5> man_1_0_2;
  bfp_adder_3(sign_1_0_0, exp_1_0_0, man_1_0_0, sign_1_0_1, exp_1_0_1,
              man_1_0_1, &sign_1_0_2, &exp_1_0_2, &man_1_0_2);

  ap_uint<8> res_1_0_exp = exp_1_0_2;
  ap_int<6> res_1_0_man = (sign_1_0_2, man_1_0_2);
  max_exp = (max_exp > exp_1_0_2) ? max_exp : exp_1_0_2;

  ap_uint<1> sign_2_0_0 = d0.data_2_0[5];
  ap_uint<8> exp_2_0_0 = d0.exponent;
  ap_uint<5> man_2_0_0 = d0.data_2_0.range(5, 0);
  ap_uint<1> sign_2_0_1 = d0.data_2_0[5];
  ap_uint<8> exp_2_0_1 = d0.exponent;
  ap_uint<5> man_2_0_1 = d0.data_2_0.range(5, 0);
  ap_uint<1> sign_2_0_2;
  ap_uint<8> exp_2_0_2;
  ap_uint<5> man_2_0_2;
  bfp_adder_3(sign_2_0_0, exp_2_0_0, man_2_0_0, sign_2_0_1, exp_2_0_1,
              man_2_0_1, &sign_2_0_2, &exp_2_0_2, &man_2_0_2);

  ap_uint<8> res_2_0_exp = exp_2_0_2;
  ap_int<6> res_2_0_man = (sign_2_0_2, man_2_0_2);
  max_exp = (max_exp > exp_2_0_2) ? max_exp : exp_2_0_2;

  ap_uint<1> sign_3_0_0 = d0.data_3_0[5];
  ap_uint<8> exp_3_0_0 = d0.exponent;
  ap_uint<5> man_3_0_0 = d0.data_3_0.range(5, 0);
  ap_uint<1> sign_3_0_1 = d0.data_3_0[5];
  ap_uint<8> exp_3_0_1 = d0.exponent;
  ap_uint<5> man_3_0_1 = d0.data_3_0.range(5, 0);
  ap_uint<1> sign_3_0_2;
  ap_uint<8> exp_3_0_2;
  ap_uint<5> man_3_0_2;
  bfp_adder_3(sign_3_0_0, exp_3_0_0, man_3_0_0, sign_3_0_1, exp_3_0_1,
              man_3_0_1, &sign_3_0_2, &exp_3_0_2, &man_3_0_2);

  ap_uint<8> res_3_0_exp = exp_3_0_2;
  ap_int<6> res_3_0_man = (sign_3_0_2, man_3_0_2);
  max_exp = (max_exp > exp_3_0_2) ? max_exp : exp_3_0_2;

  ap_uint<1> sign_4_0_0 = d0.data_4_0[5];
  ap_uint<8> exp_4_0_0 = d0.exponent;
  ap_uint<5> man_4_0_0 = d0.data_4_0.range(5, 0);
  ap_uint<1> sign_4_0_1 = d0.data_4_0[5];
  ap_uint<8> exp_4_0_1 = d0.exponent;
  ap_uint<5> man_4_0_1 = d0.data_4_0.range(5, 0);
  ap_uint<1> sign_4_0_2;
  ap_uint<8> exp_4_0_2;
  ap_uint<5> man_4_0_2;
  bfp_adder_3(sign_4_0_0, exp_4_0_0, man_4_0_0, sign_4_0_1, exp_4_0_1,
              man_4_0_1, &sign_4_0_2, &exp_4_0_2, &man_4_0_2);

  ap_uint<8> res_4_0_exp = exp_4_0_2;
  ap_int<6> res_4_0_man = (sign_4_0_2, man_4_0_2);
  max_exp = (max_exp > exp_4_0_2) ? max_exp : exp_4_0_2;

  ap_uint<1> sign_5_0_0 = d0.data_5_0[5];
  ap_uint<8> exp_5_0_0 = d0.exponent;
  ap_uint<5> man_5_0_0 = d0.data_5_0.range(5, 0);
  ap_uint<1> sign_5_0_1 = d0.data_5_0[5];
  ap_uint<8> exp_5_0_1 = d0.exponent;
  ap_uint<5> man_5_0_1 = d0.data_5_0.range(5, 0);
  ap_uint<1> sign_5_0_2;
  ap_uint<8> exp_5_0_2;
  ap_uint<5> man_5_0_2;
  bfp_adder_3(sign_5_0_0, exp_5_0_0, man_5_0_0, sign_5_0_1, exp_5_0_1,
              man_5_0_1, &sign_5_0_2, &exp_5_0_2, &man_5_0_2);

  ap_uint<8> res_5_0_exp = exp_5_0_2;
  ap_int<6> res_5_0_man = (sign_5_0_2, man_5_0_2);
  max_exp = (max_exp > exp_5_0_2) ? max_exp : exp_5_0_2;

  ap_uint<1> sign_6_0_0 = d0.data_6_0[5];
  ap_uint<8> exp_6_0_0 = d0.exponent;
  ap_uint<5> man_6_0_0 = d0.data_6_0.range(5, 0);
  ap_uint<1> sign_6_0_1 = d0.data_6_0[5];
  ap_uint<8> exp_6_0_1 = d0.exponent;
  ap_uint<5> man_6_0_1 = d0.data_6_0.range(5, 0);
  ap_uint<1> sign_6_0_2;
  ap_uint<8> exp_6_0_2;
  ap_uint<5> man_6_0_2;
  bfp_adder_3(sign_6_0_0, exp_6_0_0, man_6_0_0, sign_6_0_1, exp_6_0_1,
              man_6_0_1, &sign_6_0_2, &exp_6_0_2, &man_6_0_2);

  ap_uint<8> res_6_0_exp = exp_6_0_2;
  ap_int<6> res_6_0_man = (sign_6_0_2, man_6_0_2);
  max_exp = (max_exp > exp_6_0_2) ? max_exp : exp_6_0_2;

  ap_uint<1> sign_7_0_0 = d0.data_7_0[5];
  ap_uint<8> exp_7_0_0 = d0.exponent;
  ap_uint<5> man_7_0_0 = d0.data_7_0.range(5, 0);
  ap_uint<1> sign_7_0_1 = d0.data_7_0[5];
  ap_uint<8> exp_7_0_1 = d0.exponent;
  ap_uint<5> man_7_0_1 = d0.data_7_0.range(5, 0);
  ap_uint<1> sign_7_0_2;
  ap_uint<8> exp_7_0_2;
  ap_uint<5> man_7_0_2;
  bfp_adder_3(sign_7_0_0, exp_7_0_0, man_7_0_0, sign_7_0_1, exp_7_0_1,
              man_7_0_1, &sign_7_0_2, &exp_7_0_2, &man_7_0_2);

  ap_uint<8> res_7_0_exp = exp_7_0_2;
  ap_int<6> res_7_0_man = (sign_7_0_2, man_7_0_2);
  max_exp = (max_exp > exp_7_0_2) ? max_exp : exp_7_0_2;

  ap_uint<1> sign_8_0_0 = d0.data_8_0[5];
  ap_uint<8> exp_8_0_0 = d0.exponent;
  ap_uint<5> man_8_0_0 = d0.data_8_0.range(5, 0);
  ap_uint<1> sign_8_0_1 = d0.data_8_0[5];
  ap_uint<8> exp_8_0_1 = d0.exponent;
  ap_uint<5> man_8_0_1 = d0.data_8_0.range(5, 0);
  ap_uint<1> sign_8_0_2;
  ap_uint<8> exp_8_0_2;
  ap_uint<5> man_8_0_2;
  bfp_adder_3(sign_8_0_0, exp_8_0_0, man_8_0_0, sign_8_0_1, exp_8_0_1,
              man_8_0_1, &sign_8_0_2, &exp_8_0_2, &man_8_0_2);

  ap_uint<8> res_8_0_exp = exp_8_0_2;
  ap_int<6> res_8_0_man = (sign_8_0_2, man_8_0_2);
  max_exp = (max_exp > exp_8_0_2) ? max_exp : exp_8_0_2;

  ap_uint<1> sign_9_0_0 = d0.data_9_0[5];
  ap_uint<8> exp_9_0_0 = d0.exponent;
  ap_uint<5> man_9_0_0 = d0.data_9_0.range(5, 0);
  ap_uint<1> sign_9_0_1 = d0.data_9_0[5];
  ap_uint<8> exp_9_0_1 = d0.exponent;
  ap_uint<5> man_9_0_1 = d0.data_9_0.range(5, 0);
  ap_uint<1> sign_9_0_2;
  ap_uint<8> exp_9_0_2;
  ap_uint<5> man_9_0_2;
  bfp_adder_3(sign_9_0_0, exp_9_0_0, man_9_0_0, sign_9_0_1, exp_9_0_1,
              man_9_0_1, &sign_9_0_2, &exp_9_0_2, &man_9_0_2);

  ap_uint<8> res_9_0_exp = exp_9_0_2;
  ap_int<6> res_9_0_man = (sign_9_0_2, man_9_0_2);
  max_exp = (max_exp > exp_9_0_2) ? max_exp : exp_9_0_2;

  ap_uint<1> sign_10_0_0 = d0.data_10_0[5];
  ap_uint<8> exp_10_0_0 = d0.exponent;
  ap_uint<5> man_10_0_0 = d0.data_10_0.range(5, 0);
  ap_uint<1> sign_10_0_1 = d0.data_10_0[5];
  ap_uint<8> exp_10_0_1 = d0.exponent;
  ap_uint<5> man_10_0_1 = d0.data_10_0.range(5, 0);
  ap_uint<1> sign_10_0_2;
  ap_uint<8> exp_10_0_2;
  ap_uint<5> man_10_0_2;
  bfp_adder_3(sign_10_0_0, exp_10_0_0, man_10_0_0, sign_10_0_1, exp_10_0_1,
              man_10_0_1, &sign_10_0_2, &exp_10_0_2, &man_10_0_2);

  ap_uint<8> res_10_0_exp = exp_10_0_2;
  ap_int<6> res_10_0_man = (sign_10_0_2, man_10_0_2);
  max_exp = (max_exp > exp_10_0_2) ? max_exp : exp_10_0_2;

  ap_uint<1> sign_11_0_0 = d0.data_11_0[5];
  ap_uint<8> exp_11_0_0 = d0.exponent;
  ap_uint<5> man_11_0_0 = d0.data_11_0.range(5, 0);
  ap_uint<1> sign_11_0_1 = d0.data_11_0[5];
  ap_uint<8> exp_11_0_1 = d0.exponent;
  ap_uint<5> man_11_0_1 = d0.data_11_0.range(5, 0);
  ap_uint<1> sign_11_0_2;
  ap_uint<8> exp_11_0_2;
  ap_uint<5> man_11_0_2;
  bfp_adder_3(sign_11_0_0, exp_11_0_0, man_11_0_0, sign_11_0_1, exp_11_0_1,
              man_11_0_1, &sign_11_0_2, &exp_11_0_2, &man_11_0_2);

  ap_uint<8> res_11_0_exp = exp_11_0_2;
  ap_int<6> res_11_0_man = (sign_11_0_2, man_11_0_2);
  max_exp = (max_exp > exp_11_0_2) ? max_exp : exp_11_0_2;

  ap_uint<1> sign_12_0_0 = d0.data_12_0[5];
  ap_uint<8> exp_12_0_0 = d0.exponent;
  ap_uint<5> man_12_0_0 = d0.data_12_0.range(5, 0);
  ap_uint<1> sign_12_0_1 = d0.data_12_0[5];
  ap_uint<8> exp_12_0_1 = d0.exponent;
  ap_uint<5> man_12_0_1 = d0.data_12_0.range(5, 0);
  ap_uint<1> sign_12_0_2;
  ap_uint<8> exp_12_0_2;
  ap_uint<5> man_12_0_2;
  bfp_adder_3(sign_12_0_0, exp_12_0_0, man_12_0_0, sign_12_0_1, exp_12_0_1,
              man_12_0_1, &sign_12_0_2, &exp_12_0_2, &man_12_0_2);

  ap_uint<8> res_12_0_exp = exp_12_0_2;
  ap_int<6> res_12_0_man = (sign_12_0_2, man_12_0_2);
  max_exp = (max_exp > exp_12_0_2) ? max_exp : exp_12_0_2;

  ap_uint<1> sign_13_0_0 = d0.data_13_0[5];
  ap_uint<8> exp_13_0_0 = d0.exponent;
  ap_uint<5> man_13_0_0 = d0.data_13_0.range(5, 0);
  ap_uint<1> sign_13_0_1 = d0.data_13_0[5];
  ap_uint<8> exp_13_0_1 = d0.exponent;
  ap_uint<5> man_13_0_1 = d0.data_13_0.range(5, 0);
  ap_uint<1> sign_13_0_2;
  ap_uint<8> exp_13_0_2;
  ap_uint<5> man_13_0_2;
  bfp_adder_3(sign_13_0_0, exp_13_0_0, man_13_0_0, sign_13_0_1, exp_13_0_1,
              man_13_0_1, &sign_13_0_2, &exp_13_0_2, &man_13_0_2);

  ap_uint<8> res_13_0_exp = exp_13_0_2;
  ap_int<6> res_13_0_man = (sign_13_0_2, man_13_0_2);
  max_exp = (max_exp > exp_13_0_2) ? max_exp : exp_13_0_2;

  ap_uint<1> sign_14_0_0 = d0.data_14_0[5];
  ap_uint<8> exp_14_0_0 = d0.exponent;
  ap_uint<5> man_14_0_0 = d0.data_14_0.range(5, 0);
  ap_uint<1> sign_14_0_1 = d0.data_14_0[5];
  ap_uint<8> exp_14_0_1 = d0.exponent;
  ap_uint<5> man_14_0_1 = d0.data_14_0.range(5, 0);
  ap_uint<1> sign_14_0_2;
  ap_uint<8> exp_14_0_2;
  ap_uint<5> man_14_0_2;
  bfp_adder_3(sign_14_0_0, exp_14_0_0, man_14_0_0, sign_14_0_1, exp_14_0_1,
              man_14_0_1, &sign_14_0_2, &exp_14_0_2, &man_14_0_2);

  ap_uint<8> res_14_0_exp = exp_14_0_2;
  ap_int<6> res_14_0_man = (sign_14_0_2, man_14_0_2);
  max_exp = (max_exp > exp_14_0_2) ? max_exp : exp_14_0_2;

  ap_uint<1> sign_15_0_0 = d0.data_15_0[5];
  ap_uint<8> exp_15_0_0 = d0.exponent;
  ap_uint<5> man_15_0_0 = d0.data_15_0.range(5, 0);
  ap_uint<1> sign_15_0_1 = d0.data_15_0[5];
  ap_uint<8> exp_15_0_1 = d0.exponent;
  ap_uint<5> man_15_0_1 = d0.data_15_0.range(5, 0);
  ap_uint<1> sign_15_0_2;
  ap_uint<8> exp_15_0_2;
  ap_uint<5> man_15_0_2;
  bfp_adder_3(sign_15_0_0, exp_15_0_0, man_15_0_0, sign_15_0_1, exp_15_0_1,
              man_15_0_1, &sign_15_0_2, &exp_15_0_2, &man_15_0_2);

  ap_uint<8> res_15_0_exp = exp_15_0_2;
  ap_int<6> res_15_0_man = (sign_15_0_2, man_15_0_2);
  max_exp = (max_exp > exp_15_0_2) ? max_exp : exp_15_0_2;
  data->exponent = max_exp;
  data->data_0_0 = res_0_0_man >> (max_exp - res_0_0_exp);

  data->data_1_0 = res_1_0_man >> (max_exp - res_1_0_exp);

  data->data_2_0 = res_2_0_man >> (max_exp - res_2_0_exp);

  data->data_3_0 = res_3_0_man >> (max_exp - res_3_0_exp);

  data->data_4_0 = res_4_0_man >> (max_exp - res_4_0_exp);

  data->data_5_0 = res_5_0_man >> (max_exp - res_5_0_exp);

  data->data_6_0 = res_6_0_man >> (max_exp - res_6_0_exp);

  data->data_7_0 = res_7_0_man >> (max_exp - res_7_0_exp);

  data->data_8_0 = res_8_0_man >> (max_exp - res_8_0_exp);

  data->data_9_0 = res_9_0_man >> (max_exp - res_9_0_exp);

  data->data_10_0 = res_10_0_man >> (max_exp - res_10_0_exp);

  data->data_11_0 = res_11_0_man >> (max_exp - res_11_0_exp);

  data->data_12_0 = res_12_0_man >> (max_exp - res_12_0_exp);

  data->data_13_0 = res_13_0_man >> (max_exp - res_13_0_exp);

  data->data_14_0 = res_14_0_man >> (max_exp - res_14_0_exp);

  data->data_15_0 = res_15_0_man >> (max_exp - res_15_0_exp);
}

// BFP Add:
void bfp_add_5(hls::stream<fixed_16_1_8_5_t> &data_in_0,
               hls::stream<fixed_16_1_8_5_t> &data_in_1,
               hls::stream<fixed_16_1_8_5_t> &data_out) {
#pragma HLS INLINE OFF
  for (int j = 0; j < 688; j++) {
    for (int i = 0; i < 1; i++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II = 1
      fixed_16_1_8_5_t d0 = data_in_0.read();
      fixed_16_1_8_5_t d1 = data_in_1.read();
      fixed_16_1_8_5_t data;
      bfp_block_adder_4(d0, d1, &data);
      data_out.write(data);
    }
  }
}
