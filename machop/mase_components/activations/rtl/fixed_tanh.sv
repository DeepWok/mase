`timescale 1ns / 1ps

module fixed_tanh #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 16,  //total number of bits used to represent each input data
    parameter DATA_IN_0_PRECISION_1 = 8,  //fractional bits
    parameter DATA_IN_0_PRECISION_INT = DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1, //number of integer bits

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,  //total input data per tensor along dim 0
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,  //total input data per tensor along dim 1
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1, //input data along dim 0 coming in parallel in the same clock cycle 
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1, //input data along dim 1 coming in parallel in the same clock cycle 

    parameter DATA_OUT_0_PRECISION_0 = 16, //total number of bits used to represent each output data. Typically needs only (2 + fractional) bits since tanh varies between +/-1.
    parameter DATA_OUT_0_PRECISION_1 = 8,  //fractional bits. Output of the module is rounded to satisfy this value
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,  //total output data per tensor along dim 0
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,  //total output data per tensor along dim 1
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1, //output data along dim 0 going out in parallel in the same clock cycle
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1 //output data along dim 1 going out in parallel in the same clock cycle

) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  logic
      data_out_valid1,
      data_out_valid2,
      data_out_valid3,
      data_out_valid4; //used to store delayed version of input data valid which is given as output datavalid

  //constants a and b that divides the input range. Stored with 32 bit precision. However, they are rounded to the input precision once specified.

  const logic signed [33 : 0] a = 34'b0110000101000111101011100001010001;
  const logic signed [34 : 0] b = 35'b01010010001111010111000010100011110;
  logic signed [DATA_IN_0_PRECISION_0-1:0] a_fixed, b_fixed;

  //rounding a to input precision
  fixed_round #(
      .IN_WIDTH(34),
      .IN_FRAC_WIDTH(32),
      .OUT_WIDTH(DATA_IN_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_IN_0_PRECISION_1)
  ) fixed_round_insta (
      .data_in (a),
      .data_out(a_fixed)
  );

  //rounding b to input precision
  fixed_round #(
      .IN_WIDTH(35),
      .IN_FRAC_WIDTH(32),
      .OUT_WIDTH(DATA_IN_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_IN_0_PRECISION_1)
  ) fixed_round_instb (
      .data_in (b),
      .data_out(b_fixed)
  );

  //constants for polynomial approximation. 16 bit fractional precision is used. c1 is 1. Hence not stored.
  const logic signed [16 : 0] m1 = 17'b11011101001110111;
  const logic signed [16 : 0] d1 = 17'b00000010000011000;
  const logic signed [16 : 0] m2 = 17'b11110101001001100;
  const logic signed [16 : 0] c2 = 17'b00110110100110001;
  const logic signed [16 : 0] d2 = 17'b00111001110101111;

  //generating computation block for each parallel input data
  for (
      genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++
  ) begin : tanh
    // Local variables for computation
    logic signed [DATA_IN_0_PRECISION_0-1:0] data_in1;
    logic signed [DATA_IN_0_PRECISION_0-1:0] data_in2;
    logic signed [DATA_IN_0_PRECISION_0-1:0] data_in3;
    logic signed [DATA_IN_0_PRECISION_0-1:0] x_abs;
    logic signed [DATA_IN_0_PRECISION_0-1:0] x_abs_dum;
    logic signed [DATA_IN_0_PRECISION_0-1:0] x_abs_dum1;
    logic signed [DATA_IN_0_PRECISION_0-1:0] x_abs_dum2;
    logic signed [2*DATA_IN_0_PRECISION_0-1:0] x_squared;
    logic signed [2*DATA_IN_0_PRECISION_0-1:0] x_squared1;
    logic signed [2*DATA_IN_0_PRECISION_0-1:0] x_squared2;
    logic signed [2*DATA_IN_0_PRECISION_0+17-1:0] temp_result;
    logic signed [DATA_IN_0_PRECISION_0+17-1:0] term0;
    logic signed [2*DATA_IN_0_PRECISION_0+17-1:0] term1;
    logic signed [2*DATA_IN_0_PRECISION_0+17-1:0] term2;
    logic signed [DATA_OUT_0_PRECISION_0-1:0] temp_out;

    assign x_abs = ($signed(
            data_in_0[i]
        ) >= 0) ? data_in_0[i] : -data_in_0[i];  //calculation of absolute value

    assign x_abs_dum = x_abs;
    assign x_squared = x_abs * x_abs;  //squaring of absolute value


    always_ff @(posedge clk) begin
      if (rst) begin  //reset conditions
        term0 <= 0;
        term1 <= 0;
        term2 <= 0;
        temp_result <= 0;
      end 
			else if (data_out_0_ready && (data_in_0_valid ||data_out_valid1||data_out_valid2)) begin //Calculation of polynomial approximation.Computation is performed in two pipelined stages
        if (x_abs_dum <= a_fixed) begin
          term0 <= 0;
        end else if (x_abs_dum <= b_fixed) begin
          term0 <= c2 * x_abs;
        end else begin
          term0 <= 0;
        end
        if (x_abs_dum1 <= a_fixed) begin
          term1 <= 0;
          term1[2*DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+17-1 -1:DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+17-1] <= x_abs_dum1;
          term2 <= 0;
          term2[16+2*DATA_IN_0_PRECISION_1-1:2*DATA_IN_0_PRECISION_1] <= d1[15:0];
        end else if (x_abs_dum1 <= b_fixed) begin
          term1 <= 0;
          term1[2*DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+16-1 -1:DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+1-1] <= term0;
          term2 <= 0;
          term2[16+2*DATA_IN_0_PRECISION_1-1:2*DATA_IN_0_PRECISION_1] <= d2[15:0];
        end else begin
          term1 <= 0;
          term2 <= 0;
        end
        if (x_abs_dum2 <= a_fixed) begin
          temp_result <= m1 * x_squared2 + term1 + term2;
        end else if (x_abs_dum2 <= b_fixed) begin
          temp_result <= m2 * x_squared2 + term1 + term2;
        end else begin
          temp_result <= 1 << (2 * DATA_IN_0_PRECISION_1 + 16);
        end
      end else begin
        term0 <= 0;
        term1 <= 0;
        term2 <= 0;
        temp_result <= 0;
      end
      x_abs_dum1      <= x_abs_dum;
      x_abs_dum2      <= x_abs_dum1;
      x_squared1      <= x_squared;
      x_squared2      <= x_squared1;
      data_out_valid1 <= data_in_0_valid;
      data_out_valid2 <= data_out_valid1;
      data_out_valid3 <= data_out_valid2;
      data_in1        <= $signed(data_in_0[i]);
      data_in2        <= data_in1;
      data_in3        <= data_in2;
    end

    //rounding of the output result     
    fixed_round #(
        .IN_WIDTH(2 * DATA_IN_0_PRECISION_0 + 17),
        .IN_FRAC_WIDTH(2 * DATA_IN_0_PRECISION_1 + 16),
        .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) fixed_round_inst (
        .data_in (temp_result),
        .data_out(temp_out)
    );
    //assigning the output with sign based on sign of the input. 
    assign data_out_0[i] = (data_in3 >= 0) ? temp_out : -temp_out;

  end

  assign data_out_0_valid = data_out_valid3;
  assign data_in_0_ready  = data_out_0_ready;

endmodule

