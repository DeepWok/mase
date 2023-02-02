module dataflow_linear #(
    parameter ACT_WIDTH = 32,
    parameter W_WIDTH = 16,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUTPUT_WIDTH = 50,

    // this defines the number of elements in the vector, this is tunable
    parameter VECTOR_SIZE = 4,
    parameter NUM_VECTORS = 2,
    parameter OUTPUT_ACC_WIDTH = $clog2(NUM_VECTORS) + OUTPUT_WIDTH,

    // this defines the number of parallel instances of vector multiplication
    parameter PARALLELISM = 2,
    // parameter NUM_PARALLELLISM = 2,

    // add registers to help re-timing
    // more levels = more registers = higher fmax
    // this is a tunable parameter
    parameter VECTOR_MULT_REGISTER_LEVELS = 1,

    parameter type MY_ACT   = logic [PARALLELISM-1:0][VECTOR_SIZE-1:0][ACT_WIDTH-1:0],
    parameter type MY_W    = logic [PARALLELISM-1:0][VECTOR_SIZE-1:0][W_WIDTH-1:0],
    parameter type MY_OUTPUT  = logic [PARALLELISM-1:0][OUTPUT_ACC_WIDTH-1:0],

    // this is the type of computation to perform
    // "int" for integer, "float" for floating point
    // currently only "int" is supported
    parameter COMPUTE_TYPE = "int"
) (
    input clk,
    input rst,

    // input port for activations
    input  MY_ACT act,
    input         act_valid,
    output        act_ready,

    // input port for weights
    input MY_W weights,
    input    w_valid,
    output    w_ready,

    // output port
    output MY_OUTPUT out,
    output           out_valid,
    input            out_ready
);

  // Declare buffers for vector multiplication
  for (genvar i = 0; i < PARALLELISM; i++) begin : vector_mult_buffers
    logic [OUTPUT_WIDTH-1:0] data;
    logic valid;
    logic ready;
  end

  // Declare buffers for accumulation
  for (genvar i = 0; i < PARALLELISM; i++) begin : acccumulation_buffers
    logic [OUTPUT_ACC_WIDTH-1:0] data;
    logic valid;
    logic ready;
  end

  for (genvar i = 0; i < PARALLELISM; i = i + 1) begin
    vector_mult #(
        .ACT_WIDTH(ACT_WIDTH),
        .W_WIDTH(W_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .VECTOR_SIZE(VECTOR_SIZE),
        .REGISTER_LEVELS(VECTOR_MULT_REGISTER_LEVELS),
        .COMPUTE_TYPE(COMPUTE_TYPE)
    ) vector_mult_inst (
        .clk(clk),
        .rst(rst),
        // activation port
        .act(act[i]),
        .act_valid(act_valid),
        .act_ready(act_ready),
        // weights port
        .weights(weights[i]),
        .w_valid(w_valid),
        .w_ready(w_ready),
        // output port
        .out(vector_mult_buffers[i].data),
        .out_valid(vector_mult_buffers[i].valid),
        .out_ready(vector_mult_buffers[i].ready)
    );

    accumulator #(
        .IN_WIDTH(OUTPUT_WIDTH),
        .NUM(NUM_VECTORS)
    ) accumulator_inst (
        .clk(clk),
        .rst(rst),
        // input port
        .in(vector_mult_buffers[i].data),
        .in_valid(vector_mult_buffers[i].valid),
        .in_ready(vector_mult_buffers[i].ready),
        // output port
        .out(acccumulation_buffers[i].data),
        .out_valid(acccumulation_buffers[i].valid),
        .out_ready(acccumulation_buffers[i].ready)
    );

    assign out[i] = acccumulation_buffers[i].data;
    assign acccumulation_buffers[i].ready = out_ready;
  end

  // if one instance is valid, everything is valid, in theory
  assign out_valid = acccumulation_buffers[0].valid;

endmodule
