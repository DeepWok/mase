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
  
  logic [IN_SIZE-1:0] [(IN_WIDTH + LEVELS)-1:0] data [LEVELS:0];

  logic [LEVELS:0] valid;
  logic [LEVELS:0] ready;

  // Logic
  // ------------------------------------------------

  // Take data for first level from module interface
  for (genvar input_element = 0; input_element < IN_SIZE; input_element++) begin
    assign data  [0][input_element] = data_in[input_element];
  end
  assign valid [0] = data_in_valid;
  assign data_in_ready = ready [0];

  // Generate adder for each layer
  for (genvar level = 0; level < LEVELS; level++) begin : level
    // The number of inputs at each level = ceil(num/(2^level))
    localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << level) - 1)) >> level;
  
    // The number of adders needed at each level
    // which is the number of the inputs at next level
    localparam NEXT_LEVEL_IN_SIZE = (LEVEL_IN_SIZE + 1) / 2;
  
    localparam LEVEL_OUT_WIDTH = IN_WIDTH + level + 1;

    // The sum array is the output of the adders
    logic [LEVEL_OUT_WIDTH-1:0] sum [NEXT_LEVEL_IN_SIZE-1:0];

    logic [IN_WIDTH + level - 1 : 0] data_level_select [LEVEL_IN_SIZE-1:0];
    for (genvar j = 0; j < LEVEL_IN_SIZE; j ++) begin
      assign data_level_select[j] = data[level][j][IN_WIDTH + level - 1 : 0];
    end

    // The width of the data increases by 1 for the next
    // level in order to keep the carry bit from the addition
    fixed_adder_tree_layer #(
        .IN_SIZE (LEVEL_IN_SIZE),
        .IN_WIDTH(IN_WIDTH + level)
    ) layer (
        .data_in        (data_level_select),
        .data_in_valid  (valid[level]),
        .data_in_ready  (ready[level]),
        
        .data_out       (sum),
        .data_out_valid (valid[level+1]),
        .data_out_ready (ready[level+1])
    );

    // Drive input to next level
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin : reshape_out
      assign data[level+1][j] = sum[j];
    end

  end

  // Assign module output from last level data
  assign data_out = data[LEVELS][0];
  assign data_out_valid = valid[LEVELS];
  assign ready[LEVELS] = data_out_ready;

endmodule
