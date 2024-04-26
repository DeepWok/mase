`timescale 1ns / 1ps

module float_adder (
    input  logic [31:0] in1,
    input  logic [31:0] in2,
    output logic [31:0] res
);

  wire        [ 7:0] exp1 = in1[30:23];
  wire        [ 7:0] exp2 = in2[30:23];

  wire               s1 = exp1 > exp2 ? in1[31] : in2[31];
  wire               s2 = exp1 > exp2 ? in2[31] : in1[31];
  wire        [ 7:0] e1 = exp1 > exp2 ? in1[30:23] : in2[30:23];
  wire        [ 7:0] e2 = exp1 > exp2 ? in2[30:23] : in1[30:23];
  wire        [22:0] m1 = exp1 > exp2 ? in1[22:0] : in2[22:0];
  wire        [22:0] m2 = exp1 > exp2 ? in2[22:0] : in1[22:0];

  wire signed [24:0] f1 = {2'b01, m1};
  wire signed [24:0] f2 = {2'b01, m2};

  wire signed [25:0] sf1 = s1 ? -f1 : f1;
  wire signed [25:0] sf2 = (s2 ? -f2 : f2) >>> (e1 - e2);

  wire        [26:0] sf_sum = sf1 + sf2;
  wire        [26:0] sf_abs = sf_sum[25] ? -sf_sum : sf_sum;

  logic       [ 7:0] high_bit;
  /* verilator lint_off CASEX */
  always_comb begin
    casex ({
      sf_abs[24:0], 1'b1
    })
      26'b00000000000000000000000001: high_bit = 8'd25;
      26'b0000000000000000000000001x: high_bit = 8'd24;
      26'b000000000000000000000001xx: high_bit = 8'd23;
      26'b00000000000000000000001xxx: high_bit = 8'd22;
      26'b0000000000000000000001xxxx: high_bit = 8'd21;
      26'b000000000000000000001xxxxx: high_bit = 8'd20;
      26'b00000000000000000001xxxxxx: high_bit = 8'd19;
      26'b0000000000000000001xxxxxxx: high_bit = 8'd18;
      26'b000000000000000001xxxxxxxx: high_bit = 8'd17;
      26'b00000000000000001xxxxxxxxx: high_bit = 8'd16;
      26'b0000000000000001xxxxxxxxxx: high_bit = 8'd15;
      26'b000000000000001xxxxxxxxxxx: high_bit = 8'd14;
      26'b00000000000001xxxxxxxxxxxx: high_bit = 8'd13;
      26'b0000000000001xxxxxxxxxxxxx: high_bit = 8'd12;
      26'b000000000001xxxxxxxxxxxxxx: high_bit = 8'd11;
      26'b00000000001xxxxxxxxxxxxxxx: high_bit = 8'd10;
      26'b0000000001xxxxxxxxxxxxxxxx: high_bit = 8'd9;
      26'b000000001xxxxxxxxxxxxxxxxx: high_bit = 8'd8;
      26'b00000001xxxxxxxxxxxxxxxxxx: high_bit = 8'd7;
      26'b0000001xxxxxxxxxxxxxxxxxxx: high_bit = 8'd6;
      26'b000001xxxxxxxxxxxxxxxxxxxx: high_bit = 8'd5;
      26'b00001xxxxxxxxxxxxxxxxxxxxx: high_bit = 8'd4;
      26'b0001xxxxxxxxxxxxxxxxxxxxxx: high_bit = 8'd3;
      26'b001xxxxxxxxxxxxxxxxxxxxxxx: high_bit = 8'd2;
      26'b01xxxxxxxxxxxxxxxxxxxxxxxx: high_bit = 8'd1;
      26'b1xxxxxxxxxxxxxxxxxxxxxxxxx: high_bit = 8'd0;
      default: high_bit = 8'd26;  // should be unreachable
    endcase
  end

  wire        s = sf_sum[26];
  wire [ 7:0] e = e1 - high_bit + 1;
  wire [22:0] m = (sf_abs >> 1) << high_bit;
  wire [22:0] m_rounded = m + sf_sum[24-high_bit];
  wire [ 7:0] e_rounded = m_rounded == 0 ? e + 1 : e;

  assign res = sf_abs == 0 ? 0 : in1 == 0 ? in2 : in2 == 0 ? in1 : {s, e_rounded, m_rounded};


endmodule
