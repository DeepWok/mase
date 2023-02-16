module dot_product #(
    parameter ACT_WIDTH = 32,
    parameter W_WIDTH = 16,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUTPUT_WIDTH = ACT_WIDTH + W_WIDTH + $clog2(VECTOR_SIZE),

    // this defines the number of elements in the vector, this is tunable
    // when block arithmetics are applied, this is the same as the block size
    parameter VECTOR_SIZE = 4,

    // this is the type of computation to perform
    // "int" for integer, "float" for floating point
    // currently only "int" is supported
    parameter COMPUTE_TYPE = "int",

    // used only for block floating point
    parameter BLKFLOAT_EXP_WIDTH = 4
) (
    input clk,
    input rst,

    // input port for activations
    input  logic [ACT_WIDTH-1:0] act      [VECTOR_SIZE-1:0],
    input                        act_valid,
    output                       act_ready,

    // input port for weights
    input  logic [W_WIDTH-1:0] weights[VECTOR_SIZE-1:0],
    input                      w_valid,
    output                     w_ready,

    // output port
    output logic [OUTPUT_WIDTH-1:0] outd,
    output                          out_valid,
    input                           out_ready

    // block floating point 
    input logic [BLKFLOAT_EXP_WIDTH-1:0]    scaling,
    input                                   scaling_valid,
    output                                  scaling_ready
);

  // localparam PRODUCT_WIDTH_FLOAT = ACT_WIDTH + W_WIDTH - W_EXP_WIDTH + (1 << W_EXP_WIDTH);
  // WARNING: this is obviously wrong, just a placeholder for now
  localparam PRODUCT_WIDTH_FLOAT = ACT_WIDTH + W_WIDTH;
  localparam PRODUCT_WIDTH_FIXED = ACT_WIDTH + W_WIDTH;
  localparam PRODUCT_WIDTH = (COMPUTE_TYPE == "int") ? PRODUCT_WIDTH_FLOAT : PRODUCT_WIDTH_FIXED;


  logic [PRODUCT_WIDTH-1:0] pv       [VECTOR_SIZE-1:0];
  logic                     pv_valid;
  logic                     pv_ready;
  vector_mult #(
      .ACT_WIDTH(ACT_WIDTH),
      .W_WIDTH(W_WIDTH),
      .VECTOR_SIZE(VECTOR_SIZE),
      .COMPUTE_TYPE(COMPUTE_TYPE)
  ) vector_mult_inst (
      .clk(clk),
      .rst(rst),
      .act(act),
      .act_valid(act_valid),
      .act_ready(act_ready),
      .weights(weights),
      .w_valid(w_valid),
      .w_ready(w_ready),
      .outd(pv),
      .out_valid(pv_valid),
      .out_ready(pv_ready)
  );


  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.

  // sum the products
  logic [OUTPUT_WIDTH-1:0] sum;
  logic                    sum_valid;
  logic                    sum_ready;
  // sum = sum(pv)
  adder_tree #(
      .NUM(VECTOR_SIZE),
      .IN_WIDTH(PRODUCT_WIDTH)
  ) adder_tree_inst (
      .clk(clk),
      .rst(rst),
      .ind(pv),
      .in_valid(pv_valid),
      .in_ready(pv_ready),

      .outd(sum),
      .out_valid(sum_valid),
      .out_ready(sum_ready)
  );

  // picking the end of the buffer, wire them to the output port

  if (COMPUTE_TYPE == "int") begin
    assign outd = sum;
    assign out_valid = sum_valid;
    assign sum_ready = out_ready;
  end

endmodule
