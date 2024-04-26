`timescale 1ns / 1ps
module fixed_selu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 16,  //total number of bits used to represent each input data
    parameter DATA_IN_0_PRECISION_1 = 8,  //fractional bits
    parameter DATA_IN_0_PRECISION_INT = DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1-1, //number of integer bits

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,  //total input data per tensor along dim 0
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,  //total input data per tensor along dim 1
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1, //input data along dim 0 coming in parallel in the same clock cycle 
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,  //input data along dim 1 coming in parallel in the same clock cycle 

    parameter DATA_OUT_0_PRECISION_0 = 19, //total number of bits used to represent each output data. Typically needs only (3+ input integer + output fractional) bits due to scaling
    parameter DATA_OUT_0_PRECISION_1 = 8,  //fractional bits. Output of the module is rounded to satisfy this value
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,  //total output data per tensor along dim 0
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,  //total output data per tensor along dim 1
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,  //output data along dim 0 going out in parallel in the same clock cycle
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1, //output data along dim 1 going out in parallel in the same clock cycle

    parameter SCALE_PRECISION_1 = 16,  //fractional width of scale, max 32
    parameter ALPHA_PRECISION_1 = 16,  //fractional width of alpha, max 32

    parameter INPLACE = 0
) (
    /* verilator lint_off UNUSEDSIGNAL */
    /* verilator lint_off SELRANGE */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);
  logic signed [SCALE_PRECISION_1+1:0] scale_fixed;
  logic signed [ALPHA_PRECISION_1+1:0] alpha_fixed;

  //constants alpha and scale that of SELU. Stored with 32 bit precision. However, they are rounded to the specified precision.

  const logic signed [33:0] alpha = 34'b0110101100010110101111101011010111;
  const logic signed [33:0] scale = 34'b0100001100111110101011110101101010;

  localparam L1= SCALE_PRECISION_1+ALPHA_PRECISION_1+DATA_OUT_0_PRECISION_1+DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_1-1;
  localparam L2= SCALE_PRECISION_1+ALPHA_PRECISION_1+DATA_OUT_0_PRECISION_1-ALPHA_PRECISION_1-DATA_IN_0_PRECISION_1;

  logic
      data_out_valid1,
      data_out_valid2,
      data_out_valid3,
      data_out_valid4; //used to store delayed version of input data valid which is given as output datavalid

  //rounding scale to scale precision + 2 
  fixed_round #(
      .IN_WIDTH(34),
      .IN_FRAC_WIDTH(32),
      .OUT_WIDTH(SCALE_PRECISION_1 + 2),
      .OUT_FRAC_WIDTH(SCALE_PRECISION_1)
  ) fixed_round_inst (
      .data_in (scale),
      .data_out(scale_fixed)
  );

  //rounding alpha to alpha precision + 2 
  fixed_round #(
      .IN_WIDTH(34),
      .IN_FRAC_WIDTH(32),
      .OUT_WIDTH(ALPHA_PRECISION_1 + 2),
      .OUT_FRAC_WIDTH(ALPHA_PRECISION_1)
  ) fixed_round_inst2 (
      .data_in (alpha),
      .data_out(alpha_fixed)
  );

  logic [DATA_OUT_0_PRECISION_0-1:0] exp_out[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];


  //generating computation block for each parallel input data
  for (
      genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++
  ) begin : SeLU
    // Local variables for computation
    logic signed [DATA_IN_0_PRECISION_0-1:0] signed_data_in;
    logic signed [DATA_IN_0_PRECISION_0-1:0] signed_data_in1;
    logic signed [DATA_IN_0_PRECISION_0-1:0] signed_data_in2;
    logic signed [DATA_IN_0_PRECISION_0-1:0] signed_data_in3;
    logic [DATA_IN_0_PRECISION_0-1:0] abs_data_in;
    logic signed [DATA_OUT_0_PRECISION_0-1:0] value_1;
    logic signed [DATA_OUT_0_PRECISION_0:0] sub_value;
    logic signed [DATA_OUT_0_PRECISION_0+1+ALPHA_PRECISION_1+2-1:0] alpha_pdt;
    logic signed [DATA_OUT_0_PRECISION_0+1+ALPHA_PRECISION_1+2+SCALE_PRECISION_1+2-1:0] scale_pdt;


    assign signed_data_in = $signed(data_in_0[i]);
    assign abs_data_in = (signed_data_in <= 0) ? $unsigned(-signed_data_in) : signed_data_in;
    assign value_1[DATA_OUT_0_PRECISION_0-1:DATA_OUT_0_PRECISION_1+1] = 0;
    assign value_1[DATA_OUT_0_PRECISION_1-1:0] = 0;
    assign value_1[DATA_OUT_0_PRECISION_1] = 1;

    //calculation of negative exponential
    fixed_exp #(
        .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1 (DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
    ) exp_inst (
        .rst(rst),
        .clk(clk),
        .data_in_0(abs_data_in),
        .data_out_0(exp_out[i])
    );

    always_ff @(posedge clk) begin
      if (rst) begin  //reset conditions
        sub_value <= 0;
        alpha_pdt <= 0;
        scale_pdt <= 0;
      end 
			else if (data_out_0_ready && (data_in_0_valid ||data_out_valid1||data_out_valid2||data_out_valid3)) begin //Calculation of SELU using exponential.Computation is performed in three pipelined stages
        if (signed_data_in <= 0 || signed_data_in1 <= 0 || signed_data_in2 <= 0 || signed_data_in3 <= 0  ) begin
          sub_value <= (exp_out[i] - value_1);
          alpha_pdt <= alpha_fixed * sub_value;
          scale_pdt <= scale_fixed * alpha_pdt;
        end else begin
          sub_value <= 0;
          alpha_pdt <= 0;
          scale_pdt <= 0;
          scale_pdt[L1:L2] <= scale_fixed * signed_data_in3;
        end
      end else begin
        sub_value <= 0;
        alpha_pdt <= 0;
        scale_pdt <= 0;
      end
      signed_data_in1 <= signed_data_in;
      signed_data_in2 <= signed_data_in1;
      signed_data_in3 <= signed_data_in2;
      data_out_valid1 <= data_in_0_valid;
      data_out_valid2 <= data_out_valid1;
      data_out_valid3 <= data_out_valid2;
      data_out_valid4 <= data_out_valid3;
    end

    //rounding of the output result 
    fixed_round #(
        .IN_WIDTH(DATA_OUT_0_PRECISION_0+1+ALPHA_PRECISION_1+2+SCALE_PRECISION_1+2),            // Set the parameter values
        .IN_FRAC_WIDTH(DATA_OUT_0_PRECISION_1 + ALPHA_PRECISION_1 + SCALE_PRECISION_1),
        .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) fixed_round_inst2 (
        .data_in (scale_pdt),     // Connect inputs and outputs
        .data_out(data_out_0[i])
    );
  end

  assign data_out_0_valid = data_out_valid4;
  assign data_in_0_ready  = data_out_0_ready;

endmodule

