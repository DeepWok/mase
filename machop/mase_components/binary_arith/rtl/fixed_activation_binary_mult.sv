`timescale 1ns / 1ps
// multiply a fixed point number by -1 or 1

module fixed_activation_binary_mult #(
    parameter      IN_A_WIDTH = 32,
    parameter type TYPE_A     = logic [IN_A_WIDTH-1:0]
) (
    input TYPE_A data_a,
    input data_b,
    output TYPE_A product
);

  assign product = data_b ? data_a : -data_a;

endmodule
