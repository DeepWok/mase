`timescale 1ns / 1ps
module fixed_mlp #(
    parameter IN_WIDTH = 32,
    parameter IN_FRAC_WIDTH = 8,
    parameter WEIGHT_I2H_WIDTH = 16,
    parameter WEIGHT_I2H_FRAC_WIDTH = 8,
    parameter WEIGHT_H2O_WIDTH = 16,
    parameter WEIGHT_H2O_FRAC_WIDTH = 8,
    parameter HAS_BIAS = 1,
    parameter BIAS_I2H_WIDTH = 16,
    parameter BIAS_I2H_FRAC_WIDTH = 4,
    parameter BIAS_H2O_WIDTH = 16,
    parameter BIAS_H2O_FRAC_WIDTH = 4,
    parameter HIDDEN_WIDTH = 32,
    parameter HIDDEN_FRAC_WIDTH = 8,
    parameter OUT_WIDTH = 32,
    parameter OUT_FRAC_WIDTH = 8,

    parameter IN_NUM = 16,
    parameter IN_FEATURES = 4,
    parameter HIDDEN_FEATURES = 8,
    parameter OUT_FEATURES = IN_FEATURES,

    parameter UNROLL_IN_NUM = 4,
    parameter UNROLL_IN_FEATURES = 2,
    parameter UNROLL_HIDDEN_FEATURES = 4,
    parameter UNROLL_OUT_FEATURES = 8
) (
    input clk,
    input rst,
    //input data
    input [IN_WIDTH-1:0] data_in[UNROLL_IN_NUM * UNROLL_IN_FEATURES - 1:0],
    input data_in_valid,
    output data_in_ready,
    //input weight
    input [WEIGHT_I2H_WIDTH-1:0] weight_in2hidden[UNROLL_HIDDEN_FEATURES * UNROLL_IN_FEATURES - 1:0],
    input weight_in2hidden_valid,
    output weight_in2hidden_ready,

    input [WEIGHT_H2O_WIDTH-1:0] weight_hidden2out[UNROLL_OUT_FEATURES * UNROLL_HIDDEN_FEATURES - 1:0],
    input weight_hidden2out_valid,
    output weight_hidden2out_ready,
    //input bias
    input [BIAS_I2H_WIDTH-1:0] bias_in2hidden[UNROLL_HIDDEN_FEATURES - 1:0],
    input bias_in2hidden_valid,
    output bias_in2hidden_ready,

    input [BIAS_H2O_WIDTH-1:0] bias_hidden2out[UNROLL_OUT_FEATURES - 1:0],
    input bias_hidden2out_valid,
    output bias_hidden2out_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[UNROLL_IN_NUM * UNROLL_OUT_FEATURES - 1:0],
    output data_out_valid,
    input data_out_ready
);
  logic [HIDDEN_WIDTH-1:0] hidden_data[UNROLL_IN_NUM * UNROLL_HIDDEN_FEATURES - 1:0];
  logic hidden_data_valid, hidden_data_ready;
  logic [HIDDEN_WIDTH-1:0] relu_data[UNROLL_IN_NUM * UNROLL_HIDDEN_FEATURES - 1:0];
  logic relu_data_valid, relu_data_ready;
  fixed_2d_linear #(
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .WEIGHT_WIDTH(WEIGHT_I2H_WIDTH),
      .WEIGHT_FRAC_WIDTH(WEIGHT_I2H_FRAC_WIDTH),
      .BIAS_WIDTH(BIAS_I2H_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_I2H_FRAC_WIDTH),
      .OUT_WIDTH(HIDDEN_WIDTH),
      .OUT_FRAC_WIDTH(HIDDEN_FRAC_WIDTH),
      .IN_Y(IN_NUM),
      .IN_X(IN_FEATURES),
      .W_Y(HIDDEN_FEATURES),
      .UNROLL_IN_Y(UNROLL_IN_NUM),
      .UNROLL_IN_X(UNROLL_IN_FEATURES),
      .UNROLL_W_Y(UNROLL_HIDDEN_FEATURES)
  ) in2hidden_linear (
      .weight(weight_in2hidden),
      .weight_valid(weight_in2hidden_valid),
      .weight_ready(weight_in2hidden_ready),
      .bias(bias_in2hidden),
      .bias_valid(bias_in2hidden_valid),
      .bias_ready(bias_in2hidden_ready),
      .data_out(hidden_data),
      .data_out_valid(hidden_data_valid),
      .data_out_ready(hidden_data_ready),
      .*
  );

  fixed_relu #(
      .IN_WIDTH(HIDDEN_WIDTH),
      .IN_FRAC_WIDTH(HIDDEN_FRAC_WIDTH),
      .OUT_WIDTH(HIDDEN_WIDTH),
      .OUT_FRAC_WIDTH(HIDDEN_FRAC_WIDTH),
      .IN_SIZE(UNROLL_IN_NUM * UNROLL_HIDDEN_FEATURES)
  ) act_inst (
      .data_in(hidden_data),
      .data_in_valid(hidden_data_valid),
      .data_in_ready(hidden_data_ready),
      .data_out(relu_data),
      .data_out_valid(relu_data_valid),
      .data_out_ready(relu_data_ready),
      .*
  );

  fixed_2d_linear #(
      .IN_WIDTH(HIDDEN_WIDTH),
      .IN_FRAC_WIDTH(HIDDEN_FRAC_WIDTH),
      .WEIGHT_WIDTH(WEIGHT_H2O_WIDTH),
      .WEIGHT_FRAC_WIDTH(WEIGHT_H2O_FRAC_WIDTH),
      .BIAS_WIDTH(BIAS_H2O_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_H2O_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .IN_Y(IN_NUM),
      .IN_X(HIDDEN_FEATURES),
      .W_Y(IN_FEATURES),
      .UNROLL_IN_Y(UNROLL_IN_NUM),
      .UNROLL_IN_X(UNROLL_HIDDEN_FEATURES),
      .UNROLL_W_Y(UNROLL_OUT_FEATURES)
  ) hidden2in_linear (
      .data_in(relu_data),
      .data_in_valid(relu_data_valid),
      .data_in_ready(relu_data_ready),
      .weight(weight_hidden2out),
      .weight_valid(weight_hidden2out_valid),
      .weight_ready(weight_hidden2out_ready),
      .bias(bias_hidden2out),
      .bias_valid(bias_hidden2out_valid),
      .bias_ready(bias_hidden2out_ready),
      .*
  );
endmodule
