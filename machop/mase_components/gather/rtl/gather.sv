`timescale 1ns / 1ps


module gather #(
    parameter DIM_X = 1,
    parameter DIM_Y = 1,
    parameter PRECISION = 1

) (
    input [PRECISION-1:0] mat_a[(DIM_X*DIM_Y)-1:0],
    input [PRECISION-1:0] mat_b[(DIM_X*DIM_Y)-1:0],
    output logic [PRECISION-1:0] mat_sum[(DIM_X*DIM_Y)-1:0]
);

  // always_comb begin
    
    integer i;
    for(i=0; i<(DIM_X*DIM_Y);i=i+1) begin
      assign mat_sum = mat_a+mat_b;
    end


  // end


endmodule