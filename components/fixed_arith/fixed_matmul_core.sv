`timescale 1ns / 1ps
module fixed_matmul_core #(
    // input 
    parameter IN1_WIDTH = 32,
    parameter IN1_FRAC_WIDTH = 8,
    parameter IN2_WIDTH = 16,
    parameter IN2_FRAC_WIDTH = 8,
    parameter BIAS_WIDTH = 32,
    parameter BIAS_FRAC_WIDTH = 1,
    //output 
    parameter OUT_WIDTH = 32,
    parameter OUT_FRAC_WIDTH = 8,
    // define as nm * mk
    // rows refers to n, columns refers to m
    parameter IN1_PARALLELISM = 4,
    parameter IN_SIZE = 3,
    parameter IN2_PARALLELISM = 5,
    //defines the dataflow parameter, used for linear layer
    parameter IN_DEPTH = 3,

    parameter HAS_BIAS = 0,
    parameter BIAS_PARALLELISM = IN1_PARALLELISM,
    parameter BIAS_SIZE = IN2_PARALLELISM,
    parameter OUT_ROWS = IN1_PARALLELISM,
    parameter OUT_COLUMNS = IN2_PARALLELISM
) (
    input clk,
    input rst,
    //input data
    input [IN1_WIDTH-1:0] data_in1[IN1_PARALLELISM * IN_SIZE - 1:0],
    input data_in1_valid,
    output data_in1_ready,
    //input weight
    input [IN2_WIDTH-1:0] data_in2[IN_SIZE * IN2_PARALLELISM - 1:0],
    input data_in2_valid,
    output data_in2_ready,
    //input bias
    input [BIAS_WIDTH-1:0] bias[BIAS_PARALLELISM * BIAS_SIZE - 1:0],
    input bias_valid,
    output bias_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[OUT_ROWS * OUT_COLUMNS - 1:0],
    output data_out_valid,
    input data_out_ready
);

  // Assume the parallelised hardware above have the same arrival time
  // which means that they always have the same state. So we can just
  // pick one of the valid signal to use.
  /* verilator lint_off UNUSEDSIGNAL */
  logic fmm_join_ready, fmm_join_valid;
  logic [IN1_PARALLELISM - 1:0] fmm_data_in_ready, fmm_weight_in_ready;
  assign fmm_join_ready = fmm_data_in_ready[0];

  localparam CAST_WIDTH = IN1_WIDTH + IN2_WIDTH + $clog2(IN_SIZE * IN_DEPTH) + HAS_BIAS;
  localparam CAST_FRAC_WIDTH = IN1_FRAC_WIDTH + IN2_FRAC_WIDTH;
  logic [CAST_WIDTH-1:0] cast_data[OUT_ROWS * OUT_COLUMNS - 1:0];
  join2 #() fmm_join_inst (
      .data_in_ready ({data_in2_ready, data_in1_ready}),
      .data_in_valid ({data_in2_valid, data_in1_valid}),
      .data_out_valid(fmm_join_valid),
      .data_out_ready(fmm_join_ready)
  );


  logic [IN1_PARALLELISM - 1:0] fmm_data_out_valid;
  for (genvar i = 0; i < IN1_PARALLELISM; i++) begin : multi_linear
    // for data_in_partition
    logic [IN1_WIDTH-1:0] partition_data_in[IN_SIZE - 1:0];
    assign partition_data_in = data_in1[i*IN_SIZE+IN_SIZE-1 : i*IN_SIZE];

    logic [BIAS_WIDTH-1:0] partition_bias[IN2_PARALLELISM - 1:0];
    logic partition_bias_ready;
    assign partition_bias = bias[i*IN2_PARALLELISM+IN2_PARALLELISM-1:i*IN2_PARALLELISM];
    logic [CAST_WIDTH-1:0] fmm_data_out[IN2_PARALLELISM - 1:0];
    logic fmm_data_out_ready;
    fixed_linear #(
        .IN_WIDTH(IN1_WIDTH),
        .IN_FRAC_WIDTH(IN1_FRAC_WIDTH),
        .IN_SIZE(IN_SIZE),
        .IN_DEPTH(IN_DEPTH),
        .WEIGHT_WIDTH(IN2_WIDTH),
        .WEIGHT_FRAC_WIDTH(IN2_FRAC_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
        .PARALLELISM(IN2_PARALLELISM),
        .HAS_BIAS(HAS_BIAS)
        /* verilator lint_off PINMISSING */
    ) fl_instance (
        .clk(clk),
        .rst(rst),
        .data_in(partition_data_in),
        .data_in_valid(fmm_join_valid),
        .data_in_ready(fmm_data_in_ready[i]),
        .weight(data_in2),
        .weight_valid(fmm_join_valid),
        .weight_ready(fmm_weight_in_ready[i]),
        .bias(partition_bias),
        .bias_valid(bias_valid),
        .bias_ready(partition_bias_ready),
        .data_out(fmm_data_out),
        .data_out_valid(fmm_data_out_valid[i]),
        .data_out_ready(fmm_data_out_ready)
    );
    assign cast_data[i*OUT_COLUMNS+OUT_COLUMNS-1:i*OUT_COLUMNS] = fmm_data_out;
    assign fmm_data_out_ready = data_out_ready;
  end
  assign bias_ready = multi_linear[0].partition_bias_ready;
  assign data_out_valid = fmm_data_out_valid[0];

  fixed_cast #(
      .IN_SIZE(OUT_ROWS * OUT_COLUMNS),
      .IN_WIDTH(CAST_WIDTH),
      .IN_FRAC_WIDTH(CAST_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
  ) inst_cast (
      .data_in (cast_data),
      .data_out(data_out)
  );
endmodule
