`timescale 1ns / 1ps
module fixed_adder_tree_layer #(
    parameter IN_SIZE  = 2,
    parameter IN_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0] data_in [      IN_SIZE-1:0],
    output logic [  IN_WIDTH:0] data_out[(IN_SIZE+1)/2-1:0]
);

  generate
    for (genvar i = 0; i < IN_SIZE / 2; i++) begin : pair
      assign data_out[i] = {data_in[i][IN_WIDTH-1], data_in[i]} + {data_in[IN_SIZE-1-i][IN_WIDTH-1], data_in[IN_SIZE-1-i]};
    end

    if (IN_SIZE % 2 != 0) begin : left
      assign data_out[IN_SIZE/2] = {data_in[IN_SIZE/2][IN_WIDTH-1], data_in[IN_SIZE/2]};
    end
  endgenerate

endmodule
