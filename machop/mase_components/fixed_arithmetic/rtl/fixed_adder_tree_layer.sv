`timescale 1ns / 1ps

module fixed_adder_tree_layer #(
    parameter IN_SIZE  = 2,
    parameter IN_WIDTH = 32,

    localparam OUT_WIDTH = IN_WIDTH + 1,
    localparam OUT_SIZE = (IN_SIZE+1)/2
) (
    input  logic clk,
    input  logic rst,

    input  logic [IN_WIDTH-1:0] data_in  [IN_SIZE-1:0],
    input  logic data_in_valid,
    output logic data_in_ready,
    
    output logic [OUT_WIDTH-1:0] data_out [OUT_SIZE-1:0],
    output logic data_out_valid,
    input  logic data_out_ready
);

  localparam FLATTENED_WIDTH = OUT_SIZE * OUT_WIDTH;

  logic [OUT_SIZE-1:0] [IN_WIDTH:0] layer_out;
  logic [FLATTENED_WIDTH-1:0]       layer_out_flattened;
  logic [FLATTENED_WIDTH-1:0]       layer_out_flattened_q;

  // Calculate sum
  // ------------------------------------------------

  generate
    for (genvar i = 0; i < IN_SIZE / 2; i++) begin : pair
      // Sign extend input elements before calculating sum
      // Choose elements in "inwards" direction e.g. IN_SIZE = 8
      // -> out[0] = in[0] + in[7]
      //    out[1] = in[1] + in[6], etc
      assign layer_out[i] = {data_in[i][IN_WIDTH-1], data_in[i]} 
                            + {data_in[IN_SIZE-1-i][IN_WIDTH-1], data_in[IN_SIZE-1-i]};
    end

    // When IN_SIZE is odd, last output element is the middle element from input (sign extended)
    if (IN_SIZE % 2 != 0) begin : left
      assign layer_out[IN_SIZE/2] = {data_in[IN_SIZE/2][IN_WIDTH-1], data_in[IN_SIZE/2]};
    end
  endgenerate

  // Flatten sum before buffering
  for (genvar j = 0; j < OUT_SIZE; j++) begin : flatten
    assign layer_out_flattened [(j+1) * OUT_WIDTH - 1 : j * OUT_WIDTH] = layer_out[j];
  end : flatten

  skid_buffer #(
    .DATA_WIDTH    (FLATTENED_WIDTH)
  ) register_slice (
      .clk            (clk),
      .rst            (rst),

      .data_in        (layer_out_flattened),
      .data_in_valid  (data_in_valid),
      .data_in_ready  (data_in_ready),
      
      .data_out       (layer_out_flattened_q),
      .data_out_valid (data_out_valid),
      .data_out_ready (data_out_ready)
  );

  // Unflatten buffered sum
  for (genvar j = 0; j < OUT_SIZE; j++) begin : unflatten
    assign data_out[j] = layer_out_flattened_q[(j+1) * OUT_WIDTH - 1 : j * OUT_WIDTH];
  end : unflatten

endmodule
