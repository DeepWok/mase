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

  initial begin
    assert (2 ** LEVELS == IN_SIZE) else $fatal("IN_SIZE must be power of 2!");
  end

  logic [OUT_WIDTH*IN_SIZE-1:0] data [LEVELS:0];
  logic valid [IN_SIZE-1:0];
  logic ready [IN_SIZE-1:0];
  logic [OUT_WIDTH*IN_SIZE-1:0] sum [LEVELS-1:0];
  // logic [OUT_WIDTH-1:0] cast_sum [LEVELS-1:0];
  // logic [OUT_WIDTH-1:0] cast_data [LEVELS-1:0];

  // Declare intermediate values at each level
  // for (genvar i = 0; i <= LEVELS; i++) begin : vars
  //   // The number of inputs at each level
  //   // level_num = ceil(num/(2^i))
  //   localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;
  //   // The input data array at each level
  //   // When i = 0, data is the input of the adder tree.
  //   // When i = level, data is the output of the adder tree.
  //   logic [(IN_WIDTH + i)-1:0] data[LEVEL_IN_SIZE-1:0];
  //   // Each level has a pair of handshake signals
  //   // When i = 0, they are the handshake logic of the input.
  //   // When i = level, they are the handshake logic of the output.
  //   logic valid;
  //   logic ready;
  // end

  // Generate adder for each layer
  for (genvar i = 0; i < LEVELS; i++) begin : level

    localparam LEVEL_IN_SIZE = IN_SIZE >> i;
    localparam LEVEL_OUT_SIZE = LEVEL_IN_SIZE >> 1;
    localparam LEVEL_IN_WIDTH = IN_WIDTH + i;
    localparam LEVEL_OUT_WIDTH = LEVEL_IN_WIDTH + 1;

    // The number of adders needed at each level
    // which is the number of the inputs at next level
    // localparam NEXT_LEVEL_IN_SIZE = LEVEL_IN_SIZE >> 1;
    // The sum array is the output of the adders
    // logic [(IN_WIDTH + i):0] sum[NEXT_LEVEL_IN_SIZE-1:0];

    // The width of the data increases by 1 for the next
    // level in order to keep the carry bit from the addition
    fixed_adder_tree_layer #(
        .IN_SIZE (LEVEL_IN_SIZE),
        .IN_WIDTH(LEVEL_IN_WIDTH)
    ) layer (
        .data_in (data[i]), // flattened LEVEL_IN_SIZE * LEVEL_IN_WIDTH
        .data_out(sum[i]) // flattened LEVEL_OUT_SIZE * LEVEL_OUT_WIDTH
    );

    skid_buffer #(
        .DATA_WIDTH(LEVEL_OUT_SIZE * LEVEL_OUT_WIDTH)
    ) register_slice (
        .clk           (clk),
        .rst           (rst),
        .data_in       (sum[i]),
        .data_in_valid (valid[i]),
        .data_in_ready (ready[i]),
        .data_out      (data[i+1]),
        .data_out_valid(valid[i+1]),
        .data_out_ready(ready[i+1])
    );

    // Casting array for vars[i+1].data
    // for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin : reshape_out
    //   assign data[i+1][j] = cast_data[i][(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j];
    // end

  end

  for (genvar i = 0; i < IN_SIZE; i++) begin : gen_input_assign
    assign data[0][(i+1)*IN_WIDTH-1 : i*IN_WIDTH] = data_in[i];
  end

  assign valid[0] = data_in_valid;
  assign data_in_ready = ready[0];

  assign data_out = data[LEVELS][OUT_WIDTH-1:0];
  assign data_out_valid = valid[LEVELS];
  assign ready[LEVELS] = data_out_ready;

endmodule
