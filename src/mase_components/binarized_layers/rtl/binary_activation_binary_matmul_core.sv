`timescale 1ns / 1ps
module binary_activation_binary_matmul_core #(
    // input 
    parameter IN1_WIDTH = 1,
    parameter IN1_FRAC_WIDTH = 0,
    parameter IN2_WIDTH = 1,
    parameter IN2_FRAC_WIDTH = 0,
    //output 
    parameter OUT_WIDTH = 1,
    parameter OUT_FRAC_WIDTH = 0,
    // define as nm * mk
    // rows refers to n, columns refers to m
    parameter IN1_PARALLELISM = 4,
    parameter IN_SIZE = 3,
    parameter IN2_PARALLELISM = 5,
    //defines the dataflow parameter, used for linear layer
    parameter IN_DEPTH = 3,


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
  // Here only IN1_WIDTH is taken into account because 1 bit * N bits = N bits 
  // + 1 sign bit because the fmm_data_out is a signed number (the instance contains popcount)
  localparam CAST_WIDTH = IN1_WIDTH + $clog2(IN_SIZE * IN_DEPTH) + 1;
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

    logic [CAST_WIDTH-1:0] fmm_data_out[IN2_PARALLELISM - 1:0];
    logic fmm_data_out_ready;
    binary_activation_binary_linear #(
        .IN_WIDTH(IN1_WIDTH),
        .IN_SIZE(IN_SIZE),
        .IN_DEPTH(IN_DEPTH),
        .WEIGHT_WIDTH(IN2_WIDTH),
        .PARALLELISM(IN2_PARALLELISM),
        .HAS_BIAS(0)
        /* verilator lint_off PINMISSING */
    ) binary_activation_binary_linear_inst (
        .clk(clk),
        .rst(rst),
        .data_in(partition_data_in),
        .data_in_valid(fmm_join_valid),
        .data_in_ready(fmm_data_in_ready[i]),
        .weight(data_in2),
        .weight_valid(fmm_join_valid),
        .weight_ready(fmm_weight_in_ready[i]),
        .data_out(fmm_data_out),
        .data_out_valid(fmm_data_out_valid[i]),
        .data_out_ready(fmm_data_out_ready)
    );
    assign cast_data[i*OUT_COLUMNS+OUT_COLUMNS-1:i*OUT_COLUMNS] = fmm_data_out;
    assign fmm_data_out_ready = data_out_ready;
  end

  assign data_out_valid = fmm_data_out_valid[0];

  if (OUT_FRAC_WIDTH != 0) begin
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
  end else begin
    integer_cast #(
        .IN_SIZE(OUT_ROWS * OUT_COLUMNS),
        .IN_WIDTH(CAST_WIDTH),
        .IN_FRAC_WIDTH(CAST_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
    ) inst_cast (
        .data_in (cast_data),
        .data_out(data_out)
    );
  end
endmodule
