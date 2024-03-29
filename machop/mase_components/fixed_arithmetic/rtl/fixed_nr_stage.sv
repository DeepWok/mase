`timescale 1ns / 1ps
module fixed_nr_stage #(
    parameter WIDTH = 16,
    parameter MSB_WIDTH = 1,
    localparam THREEHALFS = 3 << (WIDTH - 2)
) (
    input  logic                 clk,
    input  logic                 rst,
    // Input x reduced
    input  logic [    WIDTH-1:0] data_a,         // FORMAT: Q1.(WIDTH-1).
    // Initial LUT guess.
    input  logic [    WIDTH-1:0] data_b,         // FORMAT: Q1.(WIDTH-1).
    input  logic [MSB_WIDTH-1:0] data_in_msb,
    input  logic                 data_in_valid,
    output logic                 data_in_ready,

    output logic [  2*WIDTH-1:0] data_out,            // FORMAT: Q1.(WIDTH-1)
    output logic [MSB_WIDTH-1:0] data_out_msb,
    output logic [    WIDTH-1:0] data_out_x_reduced,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);
  logic [2*WIDTH-1:0] yy[1:0];
  logic [WIDTH-1:0] x_reduced[3:0];
  logic [WIDTH-1:0] data_b_val[3:1];
  logic pipe_valid[3:1];
  logic pipe_ready[3:1];
  logic [2*WIDTH-1:0] mult[2:1];
  logic [2*WIDTH-1:0] threehalfs_minus_mult[3:2];
  logic [2*WIDTH-1:0] nr_out_data;
  logic [MSB_WIDTH-1:0] msb_data[3:1];

  assign yy[0] = (data_b * data_b) >> (WIDTH - 1);

  skid_buffer #(
      .DATA_WIDTH(2 * WIDTH + WIDTH + WIDTH + MSB_WIDTH)
  ) pipe_reg_0 (
      .clk(clk),
      .rst(rst),
      .data_in({yy[0], data_a, data_b, data_in_msb}),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .data_out({yy[1], x_reduced[1], data_b_val[1], msb_data[1]}),
      .data_out_valid(pipe_valid[1]),
      .data_out_ready(pipe_ready[1])
  );

  assign mult[1] = ((x_reduced[1] >> 1) * yy[1]) >> (WIDTH - 1);

  skid_buffer #(
      .DATA_WIDTH(2 * WIDTH + WIDTH + WIDTH + MSB_WIDTH)
  ) pipe_reg_1 (
      .clk(clk),
      .rst(rst),
      .data_in({mult[1], data_b_val[1], x_reduced[1], msb_data[1]}),
      .data_in_valid(pipe_valid[1]),
      .data_in_ready(pipe_ready[1]),
      .data_out({mult[2], data_b_val[2], x_reduced[2], msb_data[2]}),
      .data_out_valid(pipe_valid[2]),
      .data_out_ready(pipe_ready[2])
  );

  assign threehalfs_minus_mult[2] = THREEHALFS - mult[2];

  skid_buffer #(
      .DATA_WIDTH(2 * WIDTH + WIDTH + WIDTH + MSB_WIDTH)
  ) pipe_reg_2 (
      .clk(clk),
      .rst(rst),
      .data_in({threehalfs_minus_mult[2], data_b_val[2], x_reduced[2], msb_data[2]}),
      .data_in_valid(pipe_valid[2]),
      .data_in_ready(pipe_ready[2]),
      .data_out({threehalfs_minus_mult[3], data_b_val[3], x_reduced[3], msb_data[3]}),
      .data_out_valid(pipe_valid[3]),
      .data_out_ready(pipe_ready[3])
  );

  assign nr_out_data = (data_b_val[3] * threehalfs_minus_mult[3]) >> (WIDTH - 1);

  skid_buffer #(
      .DATA_WIDTH(2 * WIDTH + WIDTH + MSB_WIDTH)
  ) pipe_reg_3 (
      .clk(clk),
      .rst(rst),
      .data_in({nr_out_data, x_reduced[3], msb_data[3]}),
      .data_in_valid(pipe_valid[3]),
      .data_in_ready(pipe_ready[3]),
      .data_out({data_out, data_out_x_reduced, data_out_msb}),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );

endmodule
