module vector_mult #(
    parameter ACT_WIDTH = 32,
    parameter W_WIDTH = 16,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUTPUT_WIDTH = 50,

    // this defines the number of elements in the vector, this is tunable
    parameter VECTOR_SIZE = 4,

    // add registers to help re-timing
    // more levels = more registers = higher fmax
    parameter REGISTER_LEVELS = 1,

    parameter type MY_ACT   = logic [VECTOR_SIZE-1:0][ACT_WIDTH-1:0],
    parameter type MY_W    = logic [VECTOR_SIZE-1:0][W_WIDTH-1:0],
    parameter type MY_OUTPUT  = logic [OUTPUT_WIDTH-1:0],

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

  localparam type MY_PRODUCT = logic [VECTOR_SIZE-1:0][PRODUCT_WIDTH-1:0];

  // localparam PRODUCT_WIDTH_FLOAT = ACT_WIDTH + W_WIDTH - W_EXP_WIDTH + (1 << W_EXP_WIDTH);
  // WARNING: this is obviously wrong, just a placeholder for now
  localparam PRODUCT_WIDTH_FLOAT = ACT_WIDTH + W_WIDTH;
  localparam PRODUCT_WIDTH_FIXED = ACT_WIDTH + W_WIDTH;
  localparam PRODUCT_WIDTH = (COMPUTE_TYPE == "int") ? PRODUCT_WIDTH_FLOAT : PRODUCT_WIDTH_FIXED;

  logic [VECTOR_SIZE-1:0][PRODUCT_WIDTH-1:0] product_vector;

  for (genvar i = 0; i < VECTOR_SIZE; i = i + 1) begin
    if (COMPUTE_TYPE == "int") begin
      int_mult #(
          .DATA_A_WIDTH(ACT_WIDTH),
          .DATA_B_WIDTH(W_WIDTH)
      ) int_mult_inst (
          .data_a (act[i]),
          .data_b (weights[i]),
          .product(product_vector[i])
      );
    end
  end

  logic adder_tree_in_valid = act_valid & w_valid;
  logic adder_tree_in_ready;

  MY_OUTPUT sum;
  logic sum_valid, sum_ready;

  // deal with ready
  assign act_ready = adder_tree_in_ready;
  assign w_ready   = adder_tree_in_ready;

  // sum the products
  adder_tree #(
      .NUM(VECTOR_SIZE),
      .IN_WIDTH(PRODUCT_WIDTH)
  ) adder_tree_inst (
      .clk(clk),
      .rst(rst),
      .in(product_vector),
      .in_valid(adder_tree_in_valid),
      .in_ready(adder_tree_in_ready),

      .out(sum),
      .out_valid(sum_valid),
      .out_ready(sum_ready)
  );

  // Declare bufers
  for (genvar i = 0; i <= REGISTER_LEVELS; i++) begin : buffers
    logic [REGISTER_LEVELS-1:0][$bits(out)-1:0] data;
    logic valid;
    logic ready;
  end

  // Connect buffers
  assign buffers[0].data = sum;
  assign buffers[0].valid = sum_valid;
  assign sum_ready = buffers[0].ready;

  for (genvar i = 0; i < REGISTER_LEVELS; i++) begin
    register_slice #(
        .DATA_WIDTH($bits(out)),
    ) register_slice (
        .clk    (clk),
        .rst    (rst),
        .w_valid(buffers[i].valid),
        .w_ready(buffers[i].ready),
        .w_data (buffers[i].data),
        .r_valid(buffers[i+1].valid),
        .r_ready(buffers[i+1].ready),
        .r_data (buffers[i+1].data)
    );
  end

  // picking the end of the buffer, wire them to the output port
  assign out = buffers[REGISTER_LEVELS].data;
  assign out_valid = buffers[REGISTER_LEVELS].valid;
  assign buffers[REGISTER_LEVELS].ready = out_ready;

endmodule
