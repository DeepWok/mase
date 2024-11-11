`timescale 1 ns / 1 ps
module int_div #(
    parameter IN_NUM = 8,
    parameter FIFO_DEPTH = 8,
    parameter DIVIDEND_WIDTH = 8,
    parameter DIVISOR_WIDTH = 8,
    parameter QUOTIENT_WIDTH = 8
) (
    input logic clk,
    input logic rst,
    input logic [DIVIDEND_WIDTH-1:0] dividend_data[IN_NUM - 1:0],
    input logic dividend_data_valid,
    output logic dividend_data_ready,
    input logic [DIVISOR_WIDTH-1:0] divisor_data[IN_NUM - 1:0],
    input logic divisor_data_valid,
    output logic divisor_data_ready,
    output logic [QUOTIENT_WIDTH-1:0] quotient_data[IN_NUM - 1:0],
    output logic quotient_data_valid,
    input logic quotient_data_ready
);
  join2 #() join2_inst (
      .data_in_valid ({dividend_data_valid, divisor_data_valid}),
      .data_in_ready ({dividend_data_ready, divisor_data_ready}),
      .data_out_valid(quotient_data_valid),
      .data_out_ready(quotient_data_ready)
  );

  for (genvar i = 0; i < IN_NUM; i++) begin
    assign quotient_data[i] = dividend_data[i] / divisor_data[i];
  end


endmodule
