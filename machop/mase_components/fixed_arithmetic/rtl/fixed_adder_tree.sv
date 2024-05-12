`timescale 1ns / 1ps
module fixed_adder_tree #(
    parameter IN_SIZE   = 1,
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_SIZE) + IN_WIDTH
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
    fixed_adder_tree_layer #(
        .IN_SIZE (LEVEL_IN_SIZE),
        .IN_WIDTH(IN_WIDTH + i)
    ) layer (
        .data_in (vars[i].data),
        .data_out(sum)
    );

    // Cocotb/verilator does not support array flattening, so
    // we need to manually add some reshaping process.

    // Casting array for sum
    logic [$bits(sum)-1:0] cast_sum;
    logic [$bits(sum)-1:0] cast_data;
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin : reshape_in
      assign cast_sum[(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j] = sum[j];
    end

    skid_buffer #(
        .DATA_WIDTH($bits(sum))
    ) register_slice (
        .clk           (clk),
        .rst           (rst),
        .data_in       (cast_sum),
        .data_in_valid (vars[i].valid),
        .data_in_ready (vars[i].ready),
        .data_out      (cast_data),
        .data_out_valid(vars[i+1].valid),
        .data_out_ready(vars[i+1].ready)
    );

    // Casting array for vars[i+1].data
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin : reshape_out
      assign vars[i+1].data[j] = cast_data[(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j];
    end

  end

  // it will zero-extend automatically
  for (genvar j = 0; j < IN_SIZE; j++) begin : layer_0
    assign vars[0].data[j] = data_in[j];
  end
  assign vars[0].valid = data_in_valid;
  assign data_in_ready = vars[0].ready;

  assign data_out = vars[LEVELS].data[0];
  assign data_out_valid = vars[LEVELS].valid;
  assign vars[LEVELS].ready = data_out_ready;

endmodule
