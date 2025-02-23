`timescale 1ns / 1ps
module new_skid_buffer #(
    parameter DATA_WIDTH = 32
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH - 1:0] data_in,
    input  logic                    data_in_valid,
    output logic                    data_in_ready,

    output logic [DATA_WIDTH - 1:0] data_out,
    output logic                    data_out_valid,
    input  logic                    data_out_ready
);
  // feed the data_out either from
  // data_in or a buffered copy of data_in
  logic [DATA_WIDTH - 1:0] buffer;
  logic                    buffer_valid;
  logic                    buffer_ready;
  always_ff @(posedge clk) begin
    if (rst) begin
      buffer <= 0;
      buffer_valid <= 0;
    end else begin
      buffer <= data_in;
      buffer_valid <= data_in_valid;
      data_in_ready <= buffer_ready;
    end
  end
  always_comb begin
    buffer_ready = data_out_ready;
    data_out = buffer;
    data_out_valid = buffer_valid;
  end

endmodule
