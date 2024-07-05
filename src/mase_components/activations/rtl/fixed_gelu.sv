`timescale 1ns / 1ps
module fixed_gelu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    parameter IN_0_DEPTH = $rtoi($ceil(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)),

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input clk,
    input rst,

    input logic data_in_0_valid,
    output logic data_in_0_ready,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],

    output logic data_out_0_valid,
    input logic data_out_0_ready,
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]
);

  logic [DATA_IN_0_PRECISION_0-1:0] ff_data[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] roll_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];

  logic ff_data_valid;
  logic ff_data_ready;

  logic roll_data_valid;
  logic roll_data_ready;

  unpacked_fifo #(
      .DEPTH(IN_0_DEPTH),
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
  ) roller_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_0),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),  // write enable
      .data_out(ff_data),
      .data_out_valid(ff_data_valid),
      .data_out_ready(ff_data_ready)  // read enable
  );

  localparam STRAIGHT_THROUGH = (DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1 == DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1);

  generate
    if (STRAIGHT_THROUGH) begin
      unpacked_register_slice_quick #(
          .DATA_WIDTH(DATA_IN_0_PRECISION_0),
          .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
      ) single_roll (
          .clk(clk),
          .rst(rst),
          .in_data(ff_data),
          .in_valid(ff_data_valid),
          .in_ready(ff_data_ready),
          .out_data(roll_data),
          .out_valid(roll_data_valid),
          .out_ready(roll_data_ready)
      );

    end else begin

      roller #(
          .DATA_WIDTH(DATA_IN_0_PRECISION_0),
          .NUM(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
          .ROLL_NUM(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
      ) roller_inst (
          .clk(clk),
          .rst(rst),
          .data_in(ff_data),
          .data_in_valid(ff_data_valid),
          .data_in_ready(ff_data_ready),
          .data_out(roll_data),
          .data_out_valid(roll_data_valid),
          .data_out_ready(roll_data_ready)
      );
    end
  endgenerate

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : elu
    gelu_lut #(
        .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1 (DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
    ) elu_map (
        .data_in_0 (roll_data[i]),
        .data_out_0(data_out_0[i])
    );
  end

  assign data_out_0_valid = roll_data_valid;
  assign roll_data_ready  = data_out_0_ready;

endmodule
