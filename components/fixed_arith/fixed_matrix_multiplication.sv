`timescale 1ns / 1ps
module fixed_matrix_multiplication #(
    // input 
    parameter IN_WIDTH = 32,
    parameter WEIGHT_WIDTH = 16,
    // define as nm * mk
    // rows refers to n, columns refers to m
    parameter IN_ROWS = 4,
    parameter IN_COLUMNS = 3,
    parameter WEIGHT_ROWS = IN_COLUMNS,
    parameter WEIGHT_COLUMNS = 5,
    //defines the dataflow parameter, used for linear layer
    parameter IN_DEPTH = 3,

    //output 
    parameter OUT_WIDTH = IN_WIDTH + WEIGHT_WIDTH + $clog2(IN_COLUMNS) + $clog2(IN_DEPTH),

    parameter OUT_ROWS = IN_ROWS,
    parameter OUT_COLUMNS = WEIGHT_COLUMNS
) (
    input clk,
    input rst,
    //input data
    input [IN_WIDTH-1:0] data_in[IN_ROWS * IN_COLUMNS - 1:0],
    input data_in_valid,
    output data_in_ready,
    //input weight
    input [WEIGHT_WIDTH-1:0] weight[WEIGHT_ROWS * WEIGHT_COLUMNS - 1:0],
    input weight_valid,
    output weight_ready,
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
  logic [IN_ROWS - 1:0] fmm_data_in_ready, fmm_weight_in_ready;
  assign fmm_join_ready = fmm_data_in_ready[0];

  join2 #() fmm_join_inst (
      .data_in_ready ({weight_ready, data_in_ready}),
      .data_in_valid ({weight_valid, data_in_valid}),
      .data_out_valid(fmm_join_valid),
      .data_out_ready(fmm_join_ready)
  );

  for (genvar i = 0; i < IN_ROWS; i++) begin : mutil_linear
    // for data_in_partition
    logic [IN_WIDTH-1:0] partition_data_in[IN_COLUMNS - 1:0];
    assign partition_data_in = data_in[i*IN_COLUMNS+IN_COLUMNS-1 : i*IN_COLUMNS];

    logic [OUT_WIDTH-1:0] fmm_data_out[WEIGHT_COLUMNS - 1:0];
    logic fmm_data_out_valid, fmm_data_out_ready;
    fixed_linear #(
        .IN_WIDTH(IN_WIDTH),
        .IN_SIZE(IN_COLUMNS),
        .IN_DEPTH(IN_DEPTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .PARALLELISM(WEIGHT_COLUMNS),
        .HAS_BIAS(0)
        /* verilator lint_off PINMISSING */
    ) fl_instance (
        .clk(clk),
        .rst(rst),
        .data_in(partition_data_in),
        .data_in_valid(fmm_join_valid),
        .data_in_ready(fmm_data_in_ready[i]),
        .weight(weight),
        .weight_valid(fmm_join_valid),
        .weight_ready(fmm_weight_in_ready[i]),
        .data_out(fmm_data_out),
        .data_out_valid(fmm_data_out_valid),
        .data_out_ready(fmm_data_out_ready)
    );
    assign data_out[i*OUT_COLUMNS+OUT_COLUMNS-1:i*OUT_COLUMNS] = fmm_data_out;
    assign fmm_data_out_ready = data_out_ready;
  end
  assign data_out_valid = mutil_linear[0].fmm_data_out_valid;
endmodule
