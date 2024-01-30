`timescale 1ns / 1ps
// binary activation binary weight multiplier

module binary_activation_binary_mult (
    input  logic data_a,
    input  logic data_b,
    output logic product
);

  assign product = ~(data_a ^ data_b);

endmodule
