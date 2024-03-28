`timescale 1ns / 1ps
// fixed-point multiplier

module fixed_mult_quantize #(
    parameter      IN_A_WIDTH   = 16,
    parameter      IN_B_WIDTH   = 16,
    parameter type TYPE_A       = logic [           IN_A_WIDTH-1:0],
    parameter type TYPE_B       = logic [           IN_B_WIDTH-1:0],
    // parameter type TYPE_PRODUCT = logic [IN_A_WIDTH+IN_B_WIDTH-1:0]
    parameter OUT_WIDTH = 32
) (
    input TYPE_A data_a,
    input TYPE_B data_b,
    output [OUT_WIDTH-1:0] product
);

  assign product = $signed(data_a) * $signed(data_b);

endmodule