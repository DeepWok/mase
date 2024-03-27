`timescale 1ns / 1ps
module binary_activation_binary_adder_tree #(
    parameter IN_SIZE   = 1,
    // DO NOT MODIFY. This is a module for Popcount operation. 
    parameter IN_WIDTH  = 1,
    // We would need to add one more digit to the output width. Because the output is a sign number. 
    parameter OUT_WIDTH = $clog2(IN_SIZE) + IN_WIDTH + 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                 clk,
    input  logic                 rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_WIDTH-1:0] data_in       [IN_SIZE-1:0],
    input  logic                 data_in_valid,
    output logic                 data_in_ready,
    output logic [OUT_WIDTH-1:0] data_out,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);

  localparam LEVELS = $clog2(IN_SIZE);


  // Declare intermediate values at each level
  for (genvar i = 0; i <= LEVELS; i++) begin : vars
    // The number of inputs at each level
    // level_num = ceil(num/(2^i))
    localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;
    // The input data array at each level 
    // When i = 0, data is the input of the adder tree.
    // When i = level, data is the output of the adder tree.
    logic [(IN_WIDTH + i)-1:0] data[LEVEL_IN_SIZE-1:0];
    // Each level has a pair of handshake signals 
    // When i = 0, they are the handshake logic of the input.
    // When i = level, they are the handshake logic of the output.
    logic valid;
    logic ready;
  end

  // Generate adder for each layer
  for (genvar i = 0; i < LEVELS; i++) begin : level
    // The number of inputs at each level
    localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;
    // The number of adders needed at each level
    // which is the number of the inputs at next level
    localparam NEXT_LEVEL_IN_SIZE = (LEVEL_IN_SIZE + 1) / 2;
    // The sum array is the output of the adders
    logic [(IN_WIDTH + i):0] sum[NEXT_LEVEL_IN_SIZE-1:0];

    // The width of the data increases by 1 for the next
    // level in order to keep the carry bit from the addition

    binary_activation_binary_adder_tree_layer #(
        .IN_SIZE (LEVEL_IN_SIZE),
        .IN_WIDTH(IN_WIDTH + i)
    ) layer (
        .data_in (vars[i].data),
        .data_out(sum)
    );
    // always @(sum) begin
    //     $display(LEVEL_IN_SIZE);
    //     // Print all values of the vars array
    //     for (int k = 0; k < $size(vars[i].data); k++) begin
    //         $display("vars[%0d].data[%0d] = %0d", i, k, vars[i].data[k]);
    //     end

    //     for (int k = 0; k < $size(sum); k++) begin
    //         $display("sum[%0d] = %0d", k, sum[k]);
    //     end
    // end

    // Cocotb/verilator does not support array flattening, so
    // we need to manually add some reshaping process.

    // Casting array for sum
    logic [$bits(sum)-1:0] cast_sum;
    logic [$bits(sum)-1:0] cast_data;
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++)
      assign cast_sum[(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j] = sum[j];

    register_slice #(
        .IN_WIDTH($bits(sum))
    ) register_slice (
        .clk           (clk),
        .rst           (rst),
        .data_in_valid (vars[i].valid),
        .data_in_ready (vars[i].ready),
        .data_in_data  (cast_sum),
        .data_out_valid(vars[i+1].valid),
        .data_out_ready(vars[i+1].ready),
        .data_out_data (cast_data)
    );

    // Casting array for vars[i+1].data 
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin
      assign vars[i+1].data[j] = cast_data[(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j];
    end

  end

  // it will zero-extend automatically
  for (genvar j = 0; j < IN_SIZE; j++) assign vars[0].data[j] = data_in[j];
  assign vars[0].valid = data_in_valid;
  assign data_in_ready = vars[0].ready;
  assign data_out = (({1'b0, vars[LEVELS].data[0]} << 1'b1) - IN_SIZE[OUT_WIDTH-1:0]);
  assign data_out_valid = vars[LEVELS].valid;
  assign vars[LEVELS].ready = data_out_ready;

endmodule
