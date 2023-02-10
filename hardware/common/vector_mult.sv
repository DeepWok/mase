module vector_mult #(
    parameter ACT_WIDTH = 32,
    parameter W_WIDTH = 16,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUTPUT_WIDTH = ACT_WIDTH + W_WIDTH,

    // this defines the number of elements in the vector, this is tunable
    parameter VECTOR_SIZE = 4,

    // this is the type of computation to perform
    // "int" for integer, "float" for floating point
    // currently only "int" is supported
    parameter COMPUTE_TYPE = "int"
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
    output logic [OUTPUT_WIDTH-1:0] outd     [VECTOR_SIZE-1:0],
    output                          out_valid,
    input                           out_ready
);

  // localparam PRODUCT_WIDTH_FLOAT = ACT_WIDTH + W_WIDTH - W_EXP_WIDTH + (1 << W_EXP_WIDTH);
  // WARNING: this is obviously wrong, just a placeholder for now
  localparam PRODUCT_WIDTH_FLOAT = ACT_WIDTH + W_WIDTH;
  localparam PRODUCT_WIDTH_FIXED = ACT_WIDTH + W_WIDTH;
  localparam PRODUCT_WIDTH = (COMPUTE_TYPE == "int") ? PRODUCT_WIDTH_FLOAT : PRODUCT_WIDTH_FIXED;


  // pv[i] = act[i] * w[i]
  logic [PRODUCT_WIDTH-1:0] product_vector[VECTOR_SIZE-1:0];
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

  join2 #() join_inst (
      .in_ready ({w_ready, act_ready}),
      .in_valid ({w_valid, act_valid}),
      .out_valid(product_buffer_in_valid),
      .out_ready(product_buffer_in_ready)
  );

  logic product_buffer_in_valid;
  logic product_buffer_in_ready;
  logic product_buffer_out_valid;
  logic product_buffer_out_ready;

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.

  // Casting array for product vector 
  logic [$bits(product_vector)-1:0] product_vector_in;
  for (genvar i = 0; i < VECTOR_SIZE; i++)
    assign product_vector_in[PRODUCT_WIDTH*i+PRODUCT_WIDTH-1:PRODUCT_WIDTH*i] = product_vector[i];

  register_slice #(
      .DATA_WIDTH($bits(product_vector)),
  ) register_slice (
      .clk    (clk),
      .rst    (rst),
      .w_valid(product_buffer_in_valid),
      .w_ready(product_buffer_in_ready),
      .w_data (product_vector_in),
      .r_valid(product_buffer_out_valid),
      .r_ready(product_buffer_out_ready),
      .r_data (product_vector_out)
  );

  // Casting array for product vector 
  logic [$bits(product_vector)-1:0] product_vector_out;
  for (genvar i = 0; i < VECTOR_SIZE; i++)
    assign outd[i] = product_vector_out[PRODUCT_WIDTH*i+PRODUCT_WIDTH-1:PRODUCT_WIDTH*i];

  assign out_valid = product_buffer_out_valid;
  assign product_buffer_out_ready = out_ready;

endmodule
