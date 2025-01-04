`timescale 1ns / 1ps
/*
Module      : mxint_accumulator
Description : The accumulator for mxint.
              When inputing different exponent, the mantissa will cast to the same bitwidth then accumulate.
*/
module mxint_accumulator #(
    // precision_0 = mantissa_width
    // precision_1 = exponent_width
    parameter DATA_IN_0_PRECISION_0 = 4,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter UNDERFLOW_BITS = 0, // This parameter represents the number of bits that will be used to allow underflow.
    parameter BLOCK_SIZE = 4,
    parameter IN_DEPTH = 2,
    localparam DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + $clog2(IN_DEPTH) + UNDERFLOW_BITS,
    localparam DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0     [BLOCK_SIZE - 1:0],
    input  logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input  logic                             data_in_0_valid,
    output logic                             data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0     [BLOCK_SIZE - 1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic                              data_out_0_valid,
    input  logic                              data_out_0_ready
);

  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(IN_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  /* verilator lint_off WIDTH */
  assign data_in_0_ready  = (counter != IN_DEPTH) || data_out_0_ready;
  assign data_out_0_valid = (counter == IN_DEPTH);
  /* verilator lint_on WIDTH */

  localparam DATA_IN_0_PRECISION_0_EXT = DATA_IN_0_PRECISION_0 + UNDERFLOW_BITS;
  localparam DATA_OUT_0_PRECISION_0_EXT = DATA_OUT_0_PRECISION_0 + UNDERFLOW_BITS;
  // lossless shift
  logic [DATA_IN_0_PRECISION_0_EXT - 1:0] shifted_mdata_in_0[BLOCK_SIZE - 1:0];
  logic [DATA_OUT_0_PRECISION_0_EXT - 1:0] shifted_mdata_out_0[BLOCK_SIZE - 1:0];

  logic [DATA_IN_0_PRECISION_0_EXT - 1:0] extended_mdata_in_0[BLOCK_SIZE - 1:0];
  logic [DATA_OUT_0_PRECISION_0_EXT - 1:0] extended_mdata_out_0[BLOCK_SIZE - 1:0];

  logic [DATA_IN_0_PRECISION_0_EXT - 1:0] shifted_mdata_in_list [BLOCK_SIZE - 1:0][DATA_IN_0_PRECISION_0_EXT - 1:0];
  logic [DATA_OUT_0_PRECISION_0_EXT - 1:0] shifted_mdata_out_list [BLOCK_SIZE - 1:0][DATA_OUT_0_PRECISION_0_EXT - 1:0];

  logic no_value_in_register;
  logic [DATA_IN_0_PRECISION_1 - 1:0] exp_max;

  localparam SHIFT_WIDTH = DATA_IN_0_PRECISION_1 + 1;
  logic [SHIFT_WIDTH - 1:0] mdata_in_shift_value;
  logic [SHIFT_WIDTH - 1:0] mdata_in_real_shift_value;
  logic [SHIFT_WIDTH - 1:0] mdata_out_shift_value;
  logic [SHIFT_WIDTH - 1:0] mdata_out_real_shift_value;


  assign no_value_in_register =(counter == 0 || (data_out_0_valid && data_out_0_ready && data_in_0_valid));
  assign exp_max = ($signed(edata_out_0) < $signed(edata_in_0)) ? edata_in_0 : edata_out_0;
  // counter
  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (data_out_0_valid) begin
        if (data_out_0_ready) begin
          if (data_in_0_valid) counter <= 1;
          else counter <= 0;
        end
      end else if (data_in_0_valid && data_in_0_ready) counter <= counter + 1;
    end
  // mantissa
  always_comb begin
    mdata_in_shift_value = $signed(exp_max) - $signed(edata_in_0);
    mdata_out_shift_value = $signed(exp_max) - $signed(edata_out_0);
  end

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : underflow
    always_comb begin
      extended_mdata_in_0[i] = $signed(mdata_in_0[i]) <<< UNDERFLOW_BITS;
      extended_mdata_out_0[i] = $signed(mdata_out_0[i]);
    end
  end
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : optimize_variable_shift
    for (genvar j = 0; j < DATA_IN_0_PRECISION_0_EXT; j++) begin : data_in_shift
      always_comb begin
        shifted_mdata_in_list[i][j] = no_value_in_register ? $signed(extended_mdata_in_0[i]) :
            $signed(extended_mdata_in_0[i]) >>> j;
      end
    end
    for (genvar k = 0; k < DATA_OUT_0_PRECISION_0_EXT; k++) begin : data_out_shift
      always_comb begin
        shifted_mdata_out_list[i][k] = $signed(extended_mdata_out_0[i]) >>> k;
      end
    end
    assign shifted_mdata_in_0[i]  = shifted_mdata_in_list[i][mdata_in_shift_value];
    assign shifted_mdata_out_0[i] = shifted_mdata_out_list[i][mdata_out_shift_value];
  end

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : mantissa_block
    always_ff @(posedge clk)
      if (rst) mdata_out_0[i] <= '0;
      else begin
        if (data_out_0_valid) begin
          if (data_out_0_ready) begin
            if (data_in_0_valid) mdata_out_0[i] <= $signed(shifted_mdata_in_0[i]);
            else mdata_out_0[i] <= '0;
          end
        end else if (data_in_0_valid && data_in_0_ready)
          mdata_out_0[i] <= $signed(shifted_mdata_out_0[i]) + $signed(shifted_mdata_in_0[i]);
      end
  end
  localparam signed [DATA_IN_0_PRECISION_1 - 1:0] MINIMUM_EXPONENTIAL =  - 2**(DATA_IN_0_PRECISION_1 - 1);
  // exponent
  always_ff @(posedge clk)
    if (rst) edata_out_0 <= MINIMUM_EXPONENTIAL;
    else if (data_out_0_valid) begin
      if (data_out_0_ready) begin
        if (data_in_0_valid) edata_out_0 <= edata_in_0;
        else edata_out_0 <= MINIMUM_EXPONENTIAL;
      end
    end else if (data_in_0_valid && data_in_0_ready) edata_out_0 <= exp_max;

endmodule
