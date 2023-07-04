`timescale 1ns / 1ps
module padding #(
    parameter DATA_WIDTH = 32,
    parameter IMG_WIDTH = 4,
    parameter IMG_HEIGHT = 3,
    parameter PADDING_HEIGHT = 2,
    parameter PADDING_WIDTH = 2,
    parameter CHANNELS = 2
) (
    input                           clk,
    rst,
    input  logic [DATA_WIDTH - 1:0] data_in,
    input  logic                    data_in_valid,
    output logic                    data_in_ready,

    output logic [DATA_WIDTH - 1:0] data_out,
    output logic                    data_out_valid,
    input  logic                    data_out_ready
);
  localparam X_WIDTH = $clog2(PADDING_WIDTH * 2 + IMG_WIDTH) + 1;
  localparam Y_WIDTH = $clog2(PADDING_HEIGHT * 2 + IMG_HEIGHT) + 1;
  localparam C_WIDTH = $clog2(CHANNELS) + 1;

  logic [C_WIDTH -1:0] count_c;
  logic [X_WIDTH -1:0] count_x;
  logic [Y_WIDTH -1:0] count_y;
  // count position
  /* verilator lint_off WIDTH */

  logic [DATA_WIDTH - 1:0] register_out;
  logic register_out_valid, register_out_ready;
  register_slice #(
      .IN_WIDTH(DATA_WIDTH)
  ) rs_inst (
      .data_out_valid(register_out_valid),
      .data_out_ready(register_out_ready),
      .data_out_data (register_out),
      .data_in_data  (data_in),
      .*
  );
  // The start signal is used to determine whether padding output has started or ended.
  logic start;
  logic end_signal = count_c == CHANNELS - 1 
                && count_x == PADDING_WIDTH *2 + IMG_WIDTH - 1
                && count_y == PADDING_HEIGHT*2 + IMG_HEIGHT - 1
                && data_out_valid&&data_out_ready;
  always_ff @(posedge clk)
    if (rst) start <= 0;
    else if (data_in_valid && data_in_ready) start <= 1;
    else if (end_signal) start <= 0;
    else start <= start;
  always_ff @(posedge clk) begin
    if (rst) begin
      count_c <= 0;
      count_x <= 0;
      count_y <= 0;
    end else if (data_out_valid && data_out_ready)
      if(count_c == CHANNELS - 1 
                && count_x == PADDING_WIDTH *2 + IMG_WIDTH - 1
                && count_y == PADDING_HEIGHT*2 + IMG_HEIGHT - 1) begin
        count_c <= 0;
        count_x <= 0;
        count_y <= 0;
      end else if (count_c == CHANNELS - 1 && count_x == PADDING_WIDTH * 2 + IMG_WIDTH - 1) begin
        count_c <= 0;
        count_x <= 0;
        count_y <= count_y + 1;
      end else if (count_c == CHANNELS - 1) begin
        count_c <= 0;
        count_x <= count_x + 1;
        count_y <= count_y;
      end else begin
        count_c <= count_c + 1;
        count_x <= count_x;
        count_y <= count_y;
      end
  end

  logic padding_condition;
  /* verilator lint_off CMPCONST */
  /* verilator lint_off UNSIGNED */
  assign padding_condition =  (count_x < PADDING_WIDTH)
                    ||(count_x > PADDING_WIDTH + IMG_WIDTH - 1)
                    ||(count_y < PADDING_HEIGHT)
                    ||(count_y > PADDING_HEIGHT + IMG_HEIGHT - 1);
  /* verilator lint_on CMPCONST */
  /* verilator lint_on UNSIGNED */
  /* verilator lint_on WIDTH */

  assign data_out = (padding_condition) ? 0 : register_out;
  assign data_out_valid = (padding_condition) ? start : register_out_valid;
  assign register_out_ready = (padding_condition) ? 0 : data_out_ready;



endmodule
