`timescale 1ns / 1ps


module gather #(
    parameter DIM = 1,
    parameter PRECISION = 1

) (
    input [PRECISION-1:0] mat_a[DIM-1:0],
    input [PRECISION-1:0] mat_b[DIM-1:0],
    output logic [PRECISION-1:0] mat_sum[DIM-1:0]
);

  // always_comb begin
    
    integer i;
    for(i=0; i<DIM; i=i+1) begin
      assign mat_sum = mat_a + mat_b;
    end


  // end


endmodule