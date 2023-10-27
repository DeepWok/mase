`timescale 1ns / 1ps
module cut_data #(
    parameter IN_WIDTH = 32,
    parameter IN_Y = 2,
    parameter IN_X = 10,
    parameter UNROLL_IN_X = 5
) (
    input clk,
    input rst,
    input logic [IN_WIDTH - 1:0] data_in[UNROLL_IN_X -1:0],
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [IN_WIDTH - 1:0] data_out[UNROLL_IN_X -1:0],
    output logic data_out_valid,
    input logic data_out_ready
);
  logic [IN_WIDTH - 1:0] reg_in[UNROLL_IN_X - 1:0];
  logic reg_in_valid, reg_in_ready;
  localparam ITER_X = IN_X / UNROLL_IN_X;
  localparam ITER_Y = IN_Y;
  localparam Y_WIDTH = $clog2(ITER_Y);
  localparam X_WIDTH = $clog2(ITER_X);

  enum {
    PIPELINE,
    DRAIN
  } mode;
  /* verilator lint_off LITENDIAN */
  logic [Y_WIDTH-1:0] in_y;
  logic [X_WIDTH-1:0] in_x;
  /* verilator lint_on LITENDIAN */
  // data position arrange
  // always_ff @(posedge clk)
  always_ff @(posedge clk)
    if (rst) begin
      in_x <= 0;
      in_y <= 0;
      mode <= PIPELINE;
    end else begin  /* verilator lint_off WIDTH */
      if (data_in_valid && data_in_ready) begin
        if (in_y == ITER_Y - 1 && in_x == ITER_X - 1) begin
          in_x <= 0;
          in_y <= 0;
          mode <= PIPELINE;
        end else if (in_x == ITER_X - 1) begin
          in_x <= 0;
          in_y <= in_y + 1;
          mode <= DRAIN;
        end else begin
          in_x <= in_x + 1;
          in_y <= in_y;
          mode <= mode;
        end
      end
    end  /* verilator lint_on WIDTH */

  assign reg_in_valid = mode == PIPELINE && data_in_valid;
  // we can take input if our buffer is not full, or if output is ready.
  assign data_in_ready = mode == DRAIN || (mode == PIPELINE && reg_in_ready);
  assign reg_in = data_in;

  unpacked_skid_buffer #(
      .DATA_WIDTH(IN_WIDTH),
      .IN_NUM(UNROLL_IN_X)
  ) register_slice (
      .data_in_valid(reg_in_valid),
      .data_in_ready(reg_in_ready),
      .data_in(reg_in),
      .*
  );
endmodule
