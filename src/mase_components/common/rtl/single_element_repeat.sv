/*
Module      : single_element_repeat
Description : This module receives data and repeats it N times.

              This module has 2 cycle latency due to output buffering.
*/

`timescale 1ns / 1ps

module single_element_repeat #(
    parameter DATA_WIDTH = 32,
    parameter REPEAT = 2
) (
    input logic clk,
    input logic rst,

    // Input streaming port
    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    // Output streaming port
    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);

  initial begin
    assert (REPEAT > 1);
  end

  localparam CTR_WIDTH = $clog2(REPEAT);

  logic [DATA_WIDTH-1:0] output_buffer_data;
  logic output_buffer_valid, output_buffer_ready;

  typedef struct packed {
    // Data element
    logic [DATA_WIDTH-1:0] buffer_data;
    logic buffer_valid;

    // Counters
    logic [CTR_WIDTH-1:0] count;
  } SELF_T;

  SELF_T self, next_self;


  always_comb begin

    next_self = self;

    in_ready = !self.buffer_valid ||
               (self.buffer_valid && output_buffer_ready && self.count == REPEAT-1);
    output_buffer_data = self.buffer_data;
    output_buffer_valid = self.buffer_valid;

    if (in_valid && in_ready) begin
      next_self.buffer_data  = in_data;
      next_self.buffer_valid = 1;
    end

    if (output_buffer_valid && output_buffer_ready) begin
      if (self.count == REPEAT - 1) begin
        next_self.count = 0;
        if (in_valid && in_ready) begin
          next_self.buffer_data  = in_data;
          next_self.buffer_valid = 1;
        end else begin
          next_self.buffer_valid = 0;
        end
      end else begin
        next_self.count = self.count + 1;
      end
    end

  end

  skid_buffer #(
      .DATA_WIDTH(DATA_WIDTH)
  ) output_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(output_buffer_data),
      .data_in_valid(output_buffer_valid),
      .data_in_ready(output_buffer_ready),
      .data_out(out_data),
      .data_out_valid(out_valid),
      .data_out_ready(out_ready)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{default: '0};
    end else begin
      self <= next_self;
    end
  end

endmodule
