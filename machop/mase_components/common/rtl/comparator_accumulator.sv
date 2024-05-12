/*
Module      : comparator_accumulator
Description : This module implements an comparator accumulation.

              Can do signed/unsigned max/min comparisons.
*/

`timescale 1ns / 1ps

module comparator_accumulator #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 8,
    parameter MAX1_MIN0 = 1,  // MAX = 1, MIN = 0
    parameter SIGNED = 0
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);

  localparam COUNTER_WIDTH = $clog2(DEPTH) + 1;
  localparam RESET_VAL = SIGNED ? (1 << (DATA_WIDTH - 1)) : 0;

  struct {
    logic [COUNTER_WIDTH-1:0] count;
    logic [DATA_WIDTH-1:0] data;
  }
      self, next_self;

  logic [DATA_WIDTH-1:0] left, right, result;

  logic [DATA_WIDTH-1:0] output_data;
  logic output_valid, output_ready;

  // Comparator instance
  generate
    if (MAX1_MIN0) begin
      if (SIGNED) begin
        assign result = $signed(left) > $signed(right) ? left : right;
      end else begin
        assign result = left > right ? left : right;
      end
    end else begin
      if (SIGNED) begin
        assign result = $signed(left) < $signed(right) ? left : right;
      end else begin
        assign result = left < right ? left : right;
      end
    end
  endgenerate

  // Output Register Instance
  skid_buffer #(
      .DATA_WIDTH(DATA_WIDTH)
  ) out_reg (
      .clk(clk),
      .rst(rst),
      .data_in(output_data),
      .data_in_valid(output_valid),
      .data_in_ready(output_ready),
      .data_out(out_data),
      .data_out_valid(out_valid),
      .data_out_ready(out_ready)
  );


  always_comb begin
    next_self = self;

    in_ready = (self.count != DEPTH) && !((self.count == DEPTH - 1) && !output_ready);
    output_data = self.data;

    left = in_data;
    right = self.data;

    if (self.count == DEPTH) begin
      output_valid = 1;
      if (output_ready) begin
        output_data = result;
        next_self.data = RESET_VAL;
        next_self.count = 0;
      end
    end else if (self.count == DEPTH - 1) begin
      if (in_valid && in_ready) begin
        output_valid = 1;
        if (output_ready) begin
          // Redirect accumulation into outreg
          output_data = result;
          // Reset
          next_self.data = RESET_VAL;
          next_self.count = 0;
        end else begin
          next_self.count = self.count + 1;
        end
      end else begin
        output_valid = 0;
      end
    end else begin
      output_valid = 0;
      if (in_valid && in_ready) begin
        next_self.data  = result;
        next_self.count = self.count + 1;
      end
    end

  end

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{0, RESET_VAL};
    end else begin
      self <= next_self;
    end
  end


endmodule
