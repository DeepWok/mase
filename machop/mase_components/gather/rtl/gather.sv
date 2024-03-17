`timescale 1ns / 1ps


module gather #(
    parameter TENSOR_SIZE_DIM = 1,
    parameter PRECISION = 1
) (
    input [PRECISION-1:0] mat_a[TENSOR_SIZE_DIM-1:0],
    input [PRECISION-1:0] mat_b[TENSOR_SIZE_DIM-1:0],
    output logic [PRECISION-1:0] mat_sum[TENSOR_SIZE_DIM-1:0]
);

  always_comb begin
    
    integer i;
    for(i=0; i<TENSOR_SIZE_DIM; i=i+1) begin
      assign mat_sum[i] = mat_a[i] + mat_b[i];
    end

  end

endmodule