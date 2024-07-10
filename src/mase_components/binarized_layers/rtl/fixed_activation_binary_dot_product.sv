`timescale 1ns / 1ps
module fixed_activation_binary_dot_product #(
    parameter IN_WIDTH = 32,
    // this defines the number of elements in the vector, this is tunable
    // when block arithmetics are applied, this is the same as the block size
    parameter IN_SIZE = 4,
    // tuned for binary arith, DO NOT MODFIY
    parameter WEIGHT_WIDTH = 1,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUT_WIDTH = IN_WIDTH + $clog2(IN_SIZE)
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
    output logic [OUT_WIDTH-1:0] data_out,
    output                       data_out_valid,
    input                        data_out_ready

);

  localparam PRODUCT_WIDTH = IN_WIDTH;


  logic [PRODUCT_WIDTH-1:0] pv       [IN_SIZE-1:0];
  logic                     pv_valid;
  logic                     pv_ready;
  fixed_activation_binary_vector_mult #(
      .IN_WIDTH(IN_WIDTH),
      .WEIGHT_WIDTH(WEIGHT_WIDTH),
      .IN_SIZE(IN_SIZE)
  ) fixed_activation_binary_vector_mult_inst (
      .clk(clk),
      .rst(rst),
      .data_in(data_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .weight(weight),
      .weight_valid(weight_valid),
      .weight_ready(weight_ready),
      .data_out(pv),
      .data_out_valid(pv_valid),
      .data_out_ready(pv_ready)
  );


  // sum the products
  logic [OUT_WIDTH-1:0] sum;
  logic                 sum_valid;
  logic                 sum_ready;
  // sum = sum(pv)
  fixed_adder_tree #(
      .IN_SIZE (IN_SIZE),
      .IN_WIDTH(PRODUCT_WIDTH)
  ) fixed_adder_tree_inst (
      .clk(clk),
      .rst(rst),
      .data_in(pv),
      .data_in_valid(pv_valid),
      .data_in_ready(pv_ready),

      .data_out(sum),
      .data_out_valid(sum_valid),
      .data_out_ready(sum_ready)
  );

  // Picking the end of the buffer, wire them to the output port
  assign data_out = sum;
  assign data_out_valid = sum_valid;
  assign sum_ready = data_out_ready;

endmodule
