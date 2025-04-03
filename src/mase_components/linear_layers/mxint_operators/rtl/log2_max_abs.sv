`timescale 1ns / 1ps

/*
Module      : log2_max_abs
Description : Computes floor(log2(abs(input))).
              For example, if input = 4, the output will be floor(log2(abs(4))) = 2.
*/

module log2_max_abs #(
    parameter IN_SIZE   = 2,
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_WIDTH) + 1
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic [ IN_WIDTH-1:0] data_in       [IN_SIZE-1:0],
    input  logic                 data_in_valid,
    output logic                 data_in_ready,
    output logic [OUT_WIDTH-1:0] data_out,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);

  logic [IN_WIDTH - 1:0] or_result;
  logic [IN_WIDTH - 1:0] abs_data_in[IN_SIZE-1:0];

  // Compute absolute values
  generate
    for (genvar i = 0; i < IN_SIZE; i++) begin
      abs #(
          .IN_WIDTH(IN_WIDTH)
      ) abs_i (
          .data_in (data_in[i]),
          .data_out(abs_data_in[i])
      );
    end
  endgenerate

  // OR-tree to find max absolute value as only the floor(log2) is needed an OR-tree is sufficient
  or_tree #(
      .IN_SIZE (IN_SIZE),
      .IN_WIDTH(IN_WIDTH)
  ) max_bas_i (
      .clk           (clk),
      .rst           (rst),
      .data_in       (abs_data_in),
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out      (or_result),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );

  // Compute log2
  log2_value #(
      .IN_WIDTH(IN_WIDTH)
  ) log2_i (
      .data_in (or_result),
      .data_out(data_out)
  );

endmodule

/*
Module      : log2_value
Description : Computes log2 of an input value by finding the index of the highest '1' bit.
*/

module log2_value #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_WIDTH)
) (
    input  logic [IN_WIDTH - 1:0] data_in,
    output logic [ OUT_WIDTH-1:0] data_out
);

  integer i;
  logic [$clog2(IN_WIDTH)-1:0] unsigned_log_out;

  always_comb begin
    unsigned_log_out = 0;
    for (i = IN_WIDTH - 1; i >= 0; i--) begin
      if (data_in[i]) begin
        unsigned_log_out = i;
        break;
      end
    end
  end

  assign data_out = unsigned_log_out;

endmodule

/*
Module      : abs
Description : Computes the absolute value of a signed number.
*/

module abs #(
    parameter IN_WIDTH = 8
) (
    input  wire [IN_WIDTH-1:0] data_in,
    output wire [IN_WIDTH-1:0] data_out
);

  assign data_out = data_in[IN_WIDTH-1] ? ~data_in + 1 : data_in;

endmodule
