`timescale 1ns / 1ps
module fixed_adder_tree_layer #(
    parameter IN_SIZE  = 2,
    parameter IN_WIDTH = 16,
    parameter SIGNED   = 1,

    localparam OUT_WIDTH = IN_WIDTH + 1,
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
    if (SIGNED) begin
      assign data_out_unflat[i] = $signed(data_in_unflat[2*i]) + $signed(data_in_unflat[2*i+1]);
    end else begin
      assign data_out_unflat[i] = data_in_unflat[2*i] + data_in_unflat[2*i+1];
    end
  end

  if (IN_SIZE % 2 != 0) begin : left
    if (SIGNED) begin
      assign data_out_unflat[OUT_SIZE-1] = $signed(data_in_unflat[IN_SIZE-1]);
    end else begin
      assign data_out_unflat[OUT_SIZE-1] = {1'b0, data_in_unflat[IN_SIZE-1]};
    end
  end

  for (genvar i = 0; i < OUT_SIZE; i++) begin : out_flat
    assign data_out[(i+1)*OUT_WIDTH-1 : i*OUT_WIDTH] = data_out_unflat[i];
  end


endmodule
