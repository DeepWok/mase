`timescale 1ns / 1ps
module or_tree_layer #(
    parameter IN_SIZE  = 2,
    parameter IN_WIDTH = 16,

    localparam OUT_WIDTH = IN_WIDTH,
    localparam OUT_SIZE  = (IN_SIZE + 1) / 2
) (
    input  logic [  IN_SIZE*IN_WIDTH-1:0] data_in,
    output logic [OUT_SIZE*OUT_WIDTH-1:0] data_out
);

  logic [ IN_WIDTH-1:0] data_in_unflat [ IN_SIZE-1:0];
  logic [OUT_WIDTH-1:0] data_out_unflat[OUT_SIZE-1:0];

  for (genvar i = 0; i < IN_SIZE; i++) begin : in_unflat
    assign data_in_unflat[i] = data_in[(i+1)*IN_WIDTH-1 : i*IN_WIDTH];
  end

  for (genvar i = 0; i < IN_SIZE / 2; i++) begin : pair
    assign data_out_unflat[i] = data_in_unflat[2*i] | data_in_unflat[2*i+1];
  end

  if (IN_SIZE % 2 != 0) begin : left
    assign data_out_unflat[OUT_SIZE-1] = {1'b0, data_in_unflat[IN_SIZE-1]};
  end

  for (genvar i = 0; i < OUT_SIZE; i++) begin : out_flat
    assign data_out[(i+1)*OUT_WIDTH-1 : i*OUT_WIDTH] = data_out_unflat[i];
  end
endmodule


