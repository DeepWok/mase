`timescale 1ns / 1ps

module fixed_comparator_tree_layer #(
    parameter IN_SIZE  = 2,
    parameter IN_WIDTH = 16

) (

    input logic [IN_WIDTH-1:0] data_in[IN_SIZE-1:0],
    output logic [IN_WIDTH-1:0] data_out[(IN_SIZE+1)/2-1:0]
);






  generate
    for (genvar i = 0; i < IN_SIZE / 2; i++) begin : cmp_pair

      fixed_abs_comparator #(
          .IN_WIDTH(IN_WIDTH)
      ) abs_comparator_1 (
          .data_in_1(data_in[i]),
          .data_in_2(data_in[IN_SIZE-1-i]),
          .data_out (data_out[i])
      );
    end

    if (IN_SIZE % 2 != 0) begin : left
      assign data_out[IN_SIZE/2] = data_in[IN_SIZE/2];
    end
  endgenerate


endmodule
