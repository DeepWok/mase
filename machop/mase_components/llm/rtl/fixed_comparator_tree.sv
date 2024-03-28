`timescale 1ns / 1ps

module fixed_comparator_tree #(

    parameter IN_SIZE = 1,
    parameter IN_WIDTH = 16,
    parameter OUT_WIDTH = IN_WIDTH
    
    )(


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


  // Declare variables for intermediate values at each level
  for (genvar i = 0; i <= LEVELS; i++) begin : vars
    // The number of inputs at each level. 
    //It is calculated by dividing the initial number of inputs by 2^i, rounding up. 
    // This means that as we go up each level in the comparator tree, the number of inputs halves, simulating the tree structure.
    localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;

    // At level 0, the data is the input to the comparator tree.
    // At the top level, the data is the output of the comparator tree.
    logic [IN_WIDTH-1:0] data[LEVEL_IN_SIZE-1:0];

    // Each level has a pair of handshake signals to synchronize the flow of data through the comparator tree.
    // This includes a valid signal to indicate when data is available and a ready signal to indicate when the comparator at the next level is ready to receive data.

    logic valid;
    logic ready;
  end

  // Generate comparator for each layer
  for (genvar i = 0; i < LEVELS; i++) begin : level
    // The number of inputs at each level
    localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;

    // The number of comparator needed at each level,
    // which is the number of the inputs at the next level
    localparam NEXT_LEVEL_IN_SIZE = (LEVEL_IN_SIZE + 1) / 2;

    // The cmp array is the output of the comparator
    logic [IN_WIDTH-1:0] cmp [NEXT_LEVEL_IN_SIZE-1:0];

    fixed_comparator_tree_layer #(
        .IN_SIZE (LEVEL_IN_SIZE),
        .IN_WIDTH(IN_WIDTH)
    ) layer (
        .data_in (vars[i].data),
        .data_out(cmp)
    );

    // Cocotb/verilator does not support array flattening, so
    // we need to manually add some reshaping process.

    // Casting array for cmp
    logic [IN_SIZE*IN_WIDTH-1: 0] cast_cmp;
    logic [IN_SIZE*IN_WIDTH-1: 0] cast_data;
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin : reshape_in
      assign cast_cmp[IN_WIDTH*j+IN_WIDTH -1 :IN_WIDTH*j] = cmp[j];
    end

    skid_buffer #(
        .DATA_WIDTH(IN_WIDTH*IN_SIZE)
    ) register_slice (
        .clk           (clk),
        .rst           (rst),
        .data_in       (cast_cmp),
        .data_in_valid (vars[i].valid),
        .data_in_ready (vars[i].ready),
        .data_out      (cast_data),
        .data_out_valid(vars[i+1].valid),
        .data_out_ready(vars[i+1].ready)
    );

    // Casting array for vars[i+1].data
    for (genvar j = 0; j < NEXT_LEVEL_IN_SIZE; j++) begin : reshape_out
      assign vars[i+1].data[j] = cast_data[IN_WIDTH*j+IN_WIDTH-1:IN_WIDTH*j];
    end

  end



  // The following loop initializes the input layer of the comparator tree. 
  // It assigns each input data to the first level of intermediate variables, automatically performing zero-extension if necessary.

  for (genvar j = 0; j < IN_SIZE; j++) begin : layer_0
    assign vars[0].data[j] = data_in[j];
  end
  
  // This line assigns the input validity signal to the first level's valid signal, indicating that the input data is ready to be processed.
  assign vars[0].valid = data_in_valid;
  
  // The input ready signal is set based on the readiness of the first level, facilitating flow control.
  assign data_in_ready = vars[0].ready;

  // Assigns the output of the last level to the module's output. This is the result of the comparator tree.
  assign data_out = vars[LEVELS].data[0];
  
  // The validity of the module's output is determined by the last level's valid signal.
  assign data_out_valid = vars[LEVELS].valid;
  
  // The readiness to accept new output data is communicated back up the tree via the last level's ready signal.
  assign vars[LEVELS].ready = data_out_ready;


endmodule








