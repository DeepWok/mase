`timescale 1ns / 1ps
module max_abs_tree_layer #(
    parameter IN_SIZE  = 2,
    parameter IN_WIDTH = 16,

    localparam OUT_WIDTH = IN_WIDTH,
    localparam OUT_SIZE  = (IN_SIZE + 1) / 2
) (
    input  logic [  IN_SIZE*IN_WIDTH-1:0] data_in,
    output logic [OUT_SIZE*OUT_WIDTH-1:0] data_out
);

  logic [ IN_WIDTH-1:0] data_in_unflat [ IN_SIZE-1:0];
  logic [OUT_WIDTH-1:0] data_out_unflat[OUT_SIZE-1:0];

  for (genvar i = 0; i < IN_SIZE; i++) begin : in_unflat
    assign data_in_unflat[i] = data_in[(i+1)*IN_WIDTH-1 : i*IN_WIDTH];
  end

  for (genvar i = 0; i < IN_SIZE / 2; i++) begin : pair
    abs_max_value #(
        .IN_WIDTH(IN_WIDTH)
    ) abs_max_inst (
        .a  (data_in_unflat[2*i]),
        .b  (data_in_unflat[2*i+1]),
        .max(data_out_unflat[i])
    );
  end

  if (IN_SIZE % 2 != 0) begin : left
    assign data_out_unflat[OUT_SIZE-1] = {1'b0, data_in_unflat[IN_SIZE-1]};
  end

  for (genvar i = 0; i < OUT_SIZE; i++) begin : out_flat
    assign data_out[(i+1)*OUT_WIDTH-1 : i*OUT_WIDTH] = data_out_unflat[i];
  end
endmodule

module abs_max_value #(
    parameter IN_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0] a,   // 32-bit input a
    input  logic [IN_WIDTH-1:0] b,   // 32-bit input b
    output logic [IN_WIDTH-1:0] max  // 32-bit output max
);
  logic [IN_WIDTH-1:0] abs_a, abs_b;
  abs #(
      .IN_WIDTH(IN_WIDTH)
  ) abs_a_i (
      .data_in (a),
      .data_out(abs_a)
  );
  abs #(
      .IN_WIDTH(IN_WIDTH)
  ) abs_b_i (
      .data_in (b),
      .data_out(abs_b)
  );
  assign max = (abs_a > abs_b) ? abs_a : abs_b;  // If a is greater than b, max = a; otherwise, max = b
endmodule

module abs #(
    parameter IN_WIDTH = 8  // Parameter for bit-width, can be adjusted
) (
    input  wire [IN_WIDTH-1:0] data_in,  // N-bit input number
    output wire [IN_WIDTH-1:0] data_out  // N-bit output representing 2's complement
);

  // 2's complement calculation
  assign data_out = data_in[IN_WIDTH-1] ? ~data_in + 1 : data_in;
endmodule
