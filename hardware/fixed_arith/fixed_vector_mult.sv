`timescale 1ns / 1ps
module fixed_vector_mult #(
    parameter IN_WIDTH = 32,
    parameter WEIGHT_WIDTH = 16,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUT_WIDTH = IN_WIDTH + WEIGHT_WIDTH,

    // this defines the number of elements in the vector, this is tunable
    parameter IN_SIZE = 4
) (
    input clk,
    input rst,

    // input port for activations 
    input  logic [IN_WIDTH-1:0] data_in      [IN_SIZE-1:0],
    input                       data_in_valid,
    output                      data_in_ready,

    // input port for weight
    input  logic [WEIGHT_WIDTH-1:0] weight      [IN_SIZE-1:0],
    input                           weight_valid,
    output                          weight_ready,

    // output port
    output logic [OUT_WIDTH-1:0] data_out      [IN_SIZE-1:0],
    output                       data_out_valid,
    input                        data_out_ready
);

  localparam PRODUCT_WIDTH = IN_WIDTH + WEIGHT_WIDTH;

  // pv[i] = data_in[i] * w[i]
  logic [PRODUCT_WIDTH-1:0] product_vector[IN_SIZE-1:0];
  for (genvar i = 0; i < IN_SIZE; i = i + 1) begin
    fixed_mult #(
        .IN_A_WIDTH(IN_WIDTH),
        .IN_B_WIDTH(WEIGHT_WIDTH)
    ) fixed_mult_inst (
        .data_a (data_in[i]),
        .data_b (weight[i]),
        .product(product_vector[i])
    );
  end

  logic product_data_in_valid;
  logic product_data_in_ready;
  logic product_data_out_valid;
  logic product_data_out_ready;
  logic [$bits(product_vector)-1:0] product_data_in;
  logic [$bits(product_vector)-1:0] product_data_out;

  join2 #() join_inst (
      .data_in_ready ({weight_ready, data_in_ready}),
      .data_in_valid ({weight_valid, data_in_valid}),
      .data_out_valid(product_data_in_valid),
      .data_out_ready(product_data_in_ready)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.

  // Casting array for product vector 
  for (genvar i = 0; i < IN_SIZE; i++)
    assign product_data_in[PRODUCT_WIDTH*i+PRODUCT_WIDTH-1:PRODUCT_WIDTH*i] = product_vector[i];

  register_slice #(
      .IN_WIDTH($bits(product_vector))
  ) register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in_valid (product_data_in_valid),
      .data_in_ready (product_data_in_ready),
      .data_in_data  (product_data_in),
      .data_out_valid(product_data_out_valid),
      .data_out_ready(product_data_out_ready),
      .data_out_data (product_data_out)
  );

  // Casting array for product vector 
  for (genvar i = 0; i < IN_SIZE; i++)
    assign data_out[i] = product_data_out[PRODUCT_WIDTH*i+PRODUCT_WIDTH-1:PRODUCT_WIDTH*i];

  assign data_out_valid = product_data_out_valid;
  assign product_data_out_ready = data_out_ready;

endmodule
