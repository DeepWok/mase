/*
Module      : transpose
Description : This module does a combinatorial transpose of a matrix.
*/

`timescale 1ns / 1ps

module transpose #(
    parameter WIDTH = 8,
    parameter DIM0  = 4,
    parameter DIM1  = 4
) (
    input  logic [WIDTH-1:0] in_data [DIM1*DIM0-1:0],
    output logic [WIDTH-1:0] out_data[DIM1*DIM0-1:0]
);

  for (genvar i = 0; i < DIM1; i++)
  for (genvar j = 0; j < DIM0; j++) assign out_data[j*DIM1+i] = in_data[i*DIM0+j];

endmodule
