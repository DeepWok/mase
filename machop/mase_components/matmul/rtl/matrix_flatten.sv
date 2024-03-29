/*
Module      : matrix_flatten
Description : This module is a purely combinatorial block which flattens a
              row-major matrix data stream into 1D bit vector.

              Assumptions:
              data_in is already in row-major form.

              To reverse the result, you can use the "matrix_unflatten" module.
*/

`timescale 1ns / 1ps

module matrix_flatten #(
    parameter DATA_WIDTH = 32,
    parameter DIM0       = 4,
    parameter DIM1       = 4
) (
    input  logic [          DATA_WIDTH-1:0] data_in [DIM0*DIM1-1:0],
    output logic [DATA_WIDTH*DIM0*DIM1-1:0] data_out
);

  for (genvar i = 0; i < DIM0 * DIM1; i++) begin
    assign data_out[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = data_in[i];
  end

endmodule
