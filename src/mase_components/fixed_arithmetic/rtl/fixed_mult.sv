`timescale 1ns / 1ps
// fixed-point multiplier

module fixed_mult #(
    parameter      IN_A_WIDTH   = 32,
    parameter      IN_B_WIDTH   = 32,
    parameter type TYPE_A       = logic [           IN_A_WIDTH-1:0],
    parameter type TYPE_B       = logic [           IN_B_WIDTH-1:0],
    parameter type TYPE_PRODUCT = logic [IN_A_WIDTH+IN_B_WIDTH-1:0]
) (
    input TYPE_A data_a,
    input TYPE_B data_b,
    output TYPE_PRODUCT product
);

  assign product = $signed(data_a) * $signed(data_b);

endmodule
