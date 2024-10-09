`timescale 1ns / 1ps

/*
 *
*/

module fixed_linear_with_input_circular #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 1,
    parameter FIFO = 1,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    localparam IN_0_DEPTH_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    localparam IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 20,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 20,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,

    // Inferred precision of the output data
    // if the data out precision will be replaced by the setting
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DATA_IN_0_TENSOR_SIZE_DIM_0
    ) + HAS_BIAS,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,

    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = DATA_OUT_0_PARALLELISM_DIM_0,
    parameter BIAS_PARALLELISM_DIM_1 = 1
) (
    input clk,
    input rst,

    // input port for data_inivations
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // input port for weight
    input logic [WEIGHT_PRECISION_0-1:0] weight [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic weight_valid,
    output logic weight_ready,

    input logic [BIAS_PRECISION_0-1:0] bias[BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_valid,
    output logic bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
  logic [DATA_IN_0_PRECISION_0-1:0]circular_data_in_0[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic circular_data_in_0_valid, circular_data_in_0_ready;
  logic [WEIGHT_PRECISION_0-1:0]circular_weight[WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0];
  logic circular_weight_valid, circular_weight_ready;
  logic [BIAS_PRECISION_0-1:0]circular_bias[BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1-1:0];
  logic circular_bias_valid, circular_bias_ready;
  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0_reg [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic data_in_0_reg_valid, data_in_0_reg_ready;
  if (FIFO == 1) begin
    localparam FIFO_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0;
    
    fifo_for_autogen #(
        .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0), // = 8
        .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1), // = 4
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0), // = 20
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0), // = 2
        .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1), // = 4
        .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1), // = 2
        .DEPTH(FIFO_DEPTH), 
        .DATA_OUT_0_PRECISION_0(DATA_IN_0_PRECISION_0), 
        .DATA_OUT_0_PRECISION_1(DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0), 
        .DATA_OUT_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0), 
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1), 
        .DATA_OUT_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1)
    ) fifo_1_inst (
        .clk(clk),
        .rst(rst),
        .data_in_0(data_in_0),
        .data_in_0_valid(data_in_0_valid),
        .data_in_0_ready(data_in_0_ready),
        .data_out_0(data_in_0_reg),
        .data_out_0_valid(data_in_0_reg_valid),
        .data_out_0_ready(data_in_0_reg_ready)
    );
  end
  else begin
    always_comb begin
        data_in_0_reg = data_in_0;
        data_in_0_reg_valid = data_in_0_valid;
        data_in_0_ready = data_in_0_reg_ready;
    end
  end
  input_buffer #(
      .DATA_WIDTH (DATA_IN_0_PRECISION_0),
      .IN_NUM     (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
      .REPEAT     (WEIGHT_TENSOR_SIZE_DIM_1 / WEIGHT_PARALLELISM_DIM_1),
      .BUFFER_SIZE(IN_0_DEPTH_DIM_0)
  ) data_in_0_buffer (
      .clk,
      .rst,
      // Input streaming port
      .data_in(data_in_0_reg),
      .data_in_valid(data_in_0_reg_valid),
      .data_in_ready(data_in_0_reg_ready),
      // Output streaming port
      .data_out(circular_data_in_0),
      .data_out_valid(circular_data_in_0_valid),
      .data_out_ready(circular_data_in_0_ready)
  );
  input_buffer #(
      .DATA_WIDTH(WEIGHT_PRECISION_0),
      .IN_NUM(WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1),
      .REPEAT(IN_0_DEPTH_DIM_1),
      .BUFFER_SIZE(WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1 / (WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1))
  ) weight_buffer (
      .clk,
      .rst,
      // Input streaming port
      .data_in(weight),
      .data_in_valid(weight_valid),
      .data_in_ready(weight_ready),
      // Output streaming port
      .data_out(circular_weight),
      .data_out_valid(circular_weight_valid),
      .data_out_ready(circular_weight_ready)
  );
  input_buffer #(
      .DATA_WIDTH (BIAS_PRECISION_0),
      .IN_NUM     (BIAS_PARALLELISM_DIM_0),
      .REPEAT     (IN_0_DEPTH_DIM_1),
      .BUFFER_SIZE(BIAS_TENSOR_SIZE_DIM_0 / (BIAS_PARALLELISM_DIM_0))
  ) bias_buffer (
      .clk,
      .rst,
      // Input streaming port
      .data_in(bias),
      .data_in_valid(bias_valid),
      .data_in_ready(bias_ready),
      // Output streaming port
      .data_out(circular_bias),
      .data_out_valid(circular_bias_valid),
      .data_out_ready(circular_bias_ready)
  );
  logic [DATA_OUT_0_PARALLELISM_DIM_1 - 1:0]
      linear_1d_data_in_0_ready,
      linear_1d_weight_ready,
      linear_1d_bias_ready,
      linear_1d_data_out_valid;
  always_comb begin
    circular_data_in_0_ready = linear_1d_data_in_0_ready[0];
    circular_weight_ready = linear_1d_weight_ready[0];
    circular_bias_ready = linear_1d_bias_ready[0];
    data_out_0_valid = linear_1d_data_out_valid[0];
  end
  for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i = i + 1) begin
    linear_1d #(
        .HAS_BIAS(HAS_BIAS),

        .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),

        .WEIGHT_PRECISION_0      (WEIGHT_PRECISION_0),
        .WEIGHT_PRECISION_1      (WEIGHT_PRECISION_1),
        .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
        .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
        .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
        .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PARALLELISM_DIM_1),

        .BIAS_PRECISION_0      (BIAS_PRECISION_0),
        .BIAS_PRECISION_1      (BIAS_PRECISION_1),
        .BIAS_TENSOR_SIZE_DIM_0(BIAS_TENSOR_SIZE_DIM_0),
        .BIAS_PARALLELISM_DIM_0(BIAS_PARALLELISM_DIM_0),

        .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
    ) fixed_linear (
        .clk,
        .rst,
        .data_in_0      (circular_data_in_0[(i+1) * DATA_IN_0_PARALLELISM_DIM_0 - 1:i * DATA_IN_0_PARALLELISM_DIM_0]),
        .data_in_0_valid(circular_data_in_0_valid),
        .data_in_0_ready(linear_1d_data_in_0_ready[i]),
        .weight(circular_weight),
        .weight_valid(circular_weight_valid),
        .weight_ready(linear_1d_weight_ready[i]),
        .bias(circular_bias),
        .bias_valid(circular_bias_valid),
        .bias_ready(linear_1d_bias_ready[i]),
        .data_out_0      (data_out_0[(i+1) * DATA_OUT_0_PARALLELISM_DIM_0- 1:i * DATA_OUT_0_PARALLELISM_DIM_0]),
        .data_out_0_valid(linear_1d_data_out_valid[i]),
        .data_out_0_ready(data_out_0_ready)
    );
  end
endmodule
module linear_1d #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 0,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 32,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 1,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter WEIGHT_PARALLELISM_DIM_0 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1,

    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_PARALLELISM_DIM_0 = DATA_OUT_0_PARALLELISM_DIM_0

) (
    input clk,
    input rst,

    // input port for data_inivations
    input  [DATA_IN_0_PRECISION_0-1:0] data_in_0      [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input                              data_in_0_valid,
    output                             data_in_0_ready,

    // input port for weight
    input  [WEIGHT_PRECISION_0-1:0] weight      [WEIGHT_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0],
    input weight_valid,
    output weight_ready,

    /* verilator lint_off UNUSEDSIGNAL */
    input [BIAS_PRECISION_0-1:0] bias[BIAS_PARALLELISM_DIM_0-1:0],
    input bias_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output bias_ready,

    output [DATA_OUT_0_PRECISION_0-1:0] data_out_0      [DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output                              data_out_0_valid,
    input                               data_out_0_ready
);

  localparam FDP_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      DATA_IN_0_PARALLELISM_DIM_0
  );
  localparam ACC_WIDTH = FDP_WIDTH + $clog2(IN_0_DEPTH);
  localparam LOSSLESS_OUT_WIDTH = ACC_WIDTH + HAS_BIAS;
  logic fdp_join_valid, fdp_join_ready;
  join2 #() fdp_join_inst (
      .data_in_ready ({weight_ready, data_in_0_ready}),
      .data_in_valid ({weight_valid, data_in_0_valid}),
      .data_out_valid(fdp_join_valid),
      .data_out_ready(fdp_join_ready)
  );

  /* verilator lint_off UNUSEDSIGNAL */
  // Assume the parallelised hardware above have the same arrival time
  // which means that they always have the same state. So we can just
  // pick one of the valid signal to use.
  logic [WEIGHT_PARALLELISM_DIM_1-1:0] fdp_data_ready, fdp_weight_ready;
  assign fdp_join_ready = fdp_data_ready[0];
  /* verilator lint_on UNUSEDSIGNAL */

  logic                          acc_ready;
  logic [         ACC_WIDTH-1:0] acc_data_out   [DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [LOSSLESS_OUT_WIDTH-1:0] cast_data_out_0[DATA_OUT_0_PARALLELISM_DIM_0-1:0];
  // There are WEIGHT_PARALLELISM_DIM_0 number of dot product instances with DATA_IN_0_TENSOR_SIZE_DIM_0 inputs
  // and each one computes for IN_0_DEPTH iterations for each inputs.
  for (genvar i = 0; i < WEIGHT_PARALLELISM_DIM_1; i = i + 1) begin : linear
    // Assume the weight are transposed and partitioned 
    logic [WEIGHT_PRECISION_0-1:0] current_weight[DATA_IN_0_PARALLELISM_DIM_0-1:0];
    assign current_weight = weight[DATA_IN_0_PARALLELISM_DIM_0*(i+1)-1:DATA_IN_0_PARALLELISM_DIM_0*i];

    logic [FDP_WIDTH-1:0] fdp_data_out;
    logic fdp_data_out_valid, fdp_data_out_ready;

    // The inputs are already sync-ed by the previous join
    fixed_dot_product #(
        .IN_WIDTH(DATA_IN_0_PRECISION_0),
        .WEIGHT_WIDTH(WEIGHT_PRECISION_0),
        .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0)
    ) fdp_inst (
        .clk(clk),
        .rst(rst),
        .data_in(data_in_0),
        .data_in_valid(fdp_join_valid),
        .data_in_ready(fdp_data_ready[i]),
        .weight(current_weight),
        .weight_valid(fdp_join_valid),
        .weight_ready(fdp_weight_ready[i]),
        .data_out(fdp_data_out),
        .data_out_valid(fdp_data_out_valid),
        .data_out_ready(fdp_data_out_ready)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic acc_data_out_valid, acc_data_out_ready;
    /* verilator lint_on UNUSEDSIGNAL */

    fixed_accumulator #(
        .IN_WIDTH(FDP_WIDTH),
        .IN_DEPTH(IN_0_DEPTH)
    ) fixed_accumulator_inst (
        .clk(clk),
        .rst(rst),
        .data_in(fdp_data_out),
        .data_in_valid(fdp_data_out_valid),
        .data_in_ready(fdp_data_out_ready),
        .data_out(acc_data_out[i]),
        .data_out_valid(acc_data_out_valid),
        .data_out_ready(acc_data_out_ready)
    );

    // Assume the parallelised hardware above have the same arrival time
    // which means that they always have the same state. So we can just
    // pick one of the valid signal to use.
    assign acc_data_out_ready = acc_ready;
  end


  if (HAS_BIAS == 1) begin
    logic [ACC_WIDTH-1:0] bias_sext[BIAS_PARALLELISM_DIM_0-1:0];
    logic acc_join_valid, acc_join_ready;
    logic [DATA_OUT_0_PARALLELISM_DIM_0-1:0] reg_ready;

    join2 #() acc_join_inst (
        .data_in_ready ({bias_ready, acc_ready}),
        .data_in_valid ({bias_valid, linear[0].acc_data_out_valid}),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );

    fixed_rounding #(
        .IN_SIZE(DATA_OUT_0_PARALLELISM_DIM_0),
        .IN_WIDTH(BIAS_PRECISION_0),
        .IN_FRAC_WIDTH(BIAS_PRECISION_1),
        .OUT_WIDTH(ACC_WIDTH),
        .OUT_FRAC_WIDTH(DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1)
    ) bias_cast (
        .data_in (bias),
        .data_out(bias_sext)
    );

    for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_0; i = i + 1) begin : add_bias
      assign cast_data_out_0[i] = $signed(acc_data_out[i]) + $signed(bias_sext[i]);
    end
  end else begin
    assign acc_ready = data_out_0_ready;
    assign data_out_0_valid = linear[0].acc_data_out_valid;
    assign cast_data_out_0 = acc_data_out;
    assign bias_ready = 1;
  end
  fixed_rounding #(
      .IN_SIZE(DATA_OUT_0_PARALLELISM_DIM_0),
      .IN_WIDTH(LOSSLESS_OUT_WIDTH),
      .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1),
      .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  ) bias_cast (
      .data_in (cast_data_out_0),
      .data_out(data_out_0)
  );

endmodule
