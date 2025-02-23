`timescale 1ns / 1ps
module mxint_addition #(
    // precision_0 represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter DATA_IN_1_PRECISION_0 = 8,
    parameter DATA_IN_1_PRECISION_1 = 8,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    parameter DATA_IN_1_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_0 = 20,
    parameter DATA_IN_1_PARALLELISM_DIM_1 = 20,
    parameter DATA_IN_1_PARALLELISM_DIM_2 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,

    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 20,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 20,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,
    localparam BLOCK_SIZE = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1
) (
    input clk,
    input rst,
    // m -> mantissa, e -> exponent
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input data_in_0_valid,
    output data_in_0_ready,

    input logic [DATA_IN_1_PRECISION_0-1:0] mdata_in_1[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_1_PRECISION_1-1:0] edata_in_1,
    input data_in_1_valid,
    output data_in_1_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0 [BLOCK_SIZE - 1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output data_out_0_valid,
    input data_out_0_ready
);
localparam ADD_OUT_WIDTH = DATA_OUT_0_PRECISION_0 + 1;
localparam ADD_OUT_FRAC_WIDTH = DATA_OUT_0_PRECISION_0;

// Internal signals for addition pipeline
logic add_out_valid, add_out_ready;

// Signals for shift value calculation
logic [DATA_IN_0_PRECISION_1-1:0] max_value;
logic [DATA_IN_0_PRECISION_1-1:0] shift_value_0;
logic [DATA_IN_0_PRECISION_1-1:0] shift_value_1;

// Shifted mantissa signals
logic [DATA_OUT_0_PRECISION_0-1:0] shifted_mdata_in_0[BLOCK_SIZE-1:0];
logic [DATA_OUT_0_PRECISION_0-1:0] shifted_mdata_in_1[BLOCK_SIZE-1:0];

// Addition output signals
logic [ADD_OUT_WIDTH-1:0] madd_out_0[BLOCK_SIZE-1:0];
logic [DATA_IN_0_PRECISION_1-1:0] eadd_out_0;

  initial begin
    assert(
        (DATA_IN_0_PRECISION_0==DATA_IN_1_PRECISION_0) &
        (DATA_IN_0_PRECISION_1==DATA_IN_1_PRECISION_1)
    ) else $fatal("Precision of input data 0 and input data 1 must be the same");
  end


  join2 join_inst (
      .data_in_ready ({data_in_0_ready, data_in_1_ready}),
      .data_in_valid ({data_in_0_valid, data_in_1_valid}),
      .data_out_valid(add_out_valid),
      .data_out_ready(add_out_ready)
  );
  assign max_value = ($signed(edata_in_0) > $signed(edata_in_1))? edata_in_0 : edata_in_1;
  assign shift_value_0 = max_value - edata_in_0;
  assign shift_value_1 = max_value - edata_in_1;
    optimized_right_shift #(
        .IN_WIDTH(DATA_IN_0_PRECISION_0),
        .SHIFT_WIDTH(DATA_IN_0_PRECISION_1),
        .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) ovshift_0_inst (
        .data_in(mdata_in_0),
        .shift_value(shift_value),
        .data_out(shifted_mdata_in_0)
    );

    optimized_right_shift #(
        .IN_WIDTH(DATA_IN_1_PRECISION_0),
        .SHIFT_WIDTH(DATA_IN_1_PRECISION_1),
        .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) ovshift_1_inst (
        .data_in(mdata_in_1),
        .shift_value(shift_value_1),
        .data_out(shifted_mdata_in_1)
    );

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    assign madd_out_0[i] = shifted_mdata_in_0[i] + shifted_mdata_in_1[i];
  end
  assign eadd_out_0 = max_value;

  mxint_cast #(
      .IN_MAN_WIDTH(ADD_OUT_WIDTH),
      .IN_MAN_FRAC_WIDTH(ADD_OUT_FRAC_WIDTH),
      .IN_EXP_WIDTH(DATA_IN_0_PRECISION_1),
      .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) cast_i (
      .clk(clk),
      .rst(rst),
      .mdata_in(madd_out_0),  // Changed from skid_mdata_out
      .edata_in(eadd_out_0),  // Changed from skid_edata_out 
      .data_in_valid(add_out_valid),  // Changed from skid_data_out_valid
      .data_in_ready(add_out_ready),  // Changed from skid_data_out_ready
      .mdata_out(mdata_out_0),
      .edata_out(edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule
