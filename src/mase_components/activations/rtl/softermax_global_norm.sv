/*
Module      : softermax_global_norm
Description : This module implements the second section of the softermax compute
              pipeline which calculates renormalizes all local values and
              calculates the final.

              Refer to bottom half of Fig. 4.a) and 4.b) in the Softermax Paper.
              https://arxiv.org/abs/2103.09301
*/

`timescale 1ns / 1ps

module softermax_global_norm #(
    // Input shape dimensions
    parameter TOTAL_DIM   = 16,
    parameter PARALLELISM = 4,

    // Widths
    parameter IN_VALUE_WIDTH = 16,
    // IN_VALUE_FRAC_WIDTH should always be (IN_VALUE_WIDTH-1) since it is an
    // unsigned fixed-point number in range [0, 2)
    localparam IN_VALUE_FRAC_WIDTH = IN_VALUE_WIDTH - 1,
    parameter IN_MAX_WIDTH = 5,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 7
) (
    input logic clk,
    input logic rst,

    // in_values: Unsigned fixed-point in range [0, 2)
    input  logic [IN_VALUE_WIDTH-1:0] in_values[PARALLELISM-1:0],
    // in_max: Signed integers
    input  logic [  IN_MAX_WIDTH-1:0] in_max,
    input  logic                      in_valid,
    output logic                      in_ready,

    output logic [OUT_WIDTH-1:0] out_data [PARALLELISM-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  // -----
  // Parameters
  // -----

  localparam DEPTH = TOTAL_DIM / PARALLELISM;

  // Max is integer only
  localparam SUBTRACT_WIDTH = IN_MAX_WIDTH + 1;

  localparam ADDER_TREE_IN_WIDTH = 1 + IN_VALUE_WIDTH;  // Pad single zero for unsigned
  localparam ADDER_TREE_OUT_WIDTH = $clog2(PARALLELISM) + ADDER_TREE_IN_WIDTH;
  localparam ADDER_TREE_FRAC_WIDTH = IN_VALUE_FRAC_WIDTH;

  localparam ACC_WIDTH = $clog2(DEPTH) + ADDER_TREE_OUT_WIDTH;
  localparam ACC_FRAC_WIDTH = ADDER_TREE_FRAC_WIDTH;

  // TODO: Maybe set this at top level?
  localparam RECIP_WIDTH = 2 * ACC_WIDTH;
  localparam RECIP_FRAC_WIDTH = 2 * ACC_FRAC_WIDTH;

  localparam MULT_WIDTH = IN_VALUE_WIDTH + RECIP_WIDTH;
  localparam MULT_FRAC_WIDTH = IN_VALUE_FRAC_WIDTH + RECIP_FRAC_WIDTH;


  initial begin
    assert (TOTAL_DIM > 1);
    assert (DEPTH * PARALLELISM == TOTAL_DIM);

    // Sanity Check
    assert (ADDER_TREE_OUT_WIDTH >= ADDER_TREE_FRAC_WIDTH);
    assert (ADDER_TREE_IN_WIDTH >= ADDER_TREE_FRAC_WIDTH);
    assert (ACC_WIDTH >= ACC_FRAC_WIDTH);
    assert (RECIP_WIDTH >= RECIP_FRAC_WIDTH);
    assert (MULT_WIDTH >= MULT_FRAC_WIDTH);
  end


  // -----
  // Wires
  // -----

  logic [IN_MAX_WIDTH-1:0] local_max_out;
  logic local_max_in_valid, local_max_in_ready;
  logic local_max_out_valid, local_max_out_ready;

  logic [IN_MAX_WIDTH-1:0] global_max_out;
  logic global_max_in_valid, global_max_in_ready;
  logic global_max_out_valid, global_max_out_ready;

  logic [IN_MAX_WIDTH-1:0] repeat_global_max_out;
  logic repeat_global_max_out_valid, repeat_global_max_out_ready;

  logic [IN_VALUE_WIDTH-1:0] local_values_out[PARALLELISM-1:0];
  logic local_values_in_valid, local_values_in_ready;
  logic local_values_out_valid, local_values_out_ready;

  logic [SUBTRACT_WIDTH-1:0] subtract_in_data;
  logic [SUBTRACT_WIDTH-1:0] subtract_out_data;
  logic subtract_in_valid, subtract_in_ready;
  logic subtract_out_valid, subtract_out_ready;

  logic [IN_VALUE_WIDTH-1:0] shift_in_data[PARALLELISM-1:0];
  logic [IN_VALUE_WIDTH-1:0] shift_out_data[PARALLELISM-1:0];
  logic shift_in_valid;
  logic shift_in_ready[PARALLELISM-1:0];
  logic shift_out_valid[PARALLELISM-1:0];
  logic shift_out_ready;

  logic [IN_VALUE_WIDTH-1:0] adjusted_values_in_data[PARALLELISM-1:0];
  logic [IN_VALUE_WIDTH-1:0] adjusted_values_out_data[PARALLELISM-1:0];
  logic adjusted_values_in_valid, adjusted_values_in_ready;
  logic adjusted_values_out_valid, adjusted_values_out_ready;

  logic [ ADDER_TREE_IN_WIDTH-1:0] adder_tree_in_data  [PARALLELISM-1:0];
  logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_out_data;
  logic adder_tree_in_valid, adder_tree_in_ready;
  logic adder_tree_out_valid, adder_tree_out_ready;

  logic [ACC_WIDTH-1:0] acc_out_data;
  logic acc_out_valid, acc_out_ready;

  logic [RECIP_WIDTH-1:0] norm_recip_data;
  logic norm_recip_valid, norm_recip_ready;

  logic [RECIP_WIDTH-1:0] repeat_norm_recip_data;
  logic repeat_norm_recip_valid, repeat_norm_recip_ready;

  logic [MULT_WIDTH-1:0] mult_in_data[PARALLELISM-1:0];
  logic [MULT_WIDTH-1:0] mult_out_data[PARALLELISM-1:0];
  logic mult_in_valid;
  logic mult_in_ready[PARALLELISM-1:0];
  logic mult_out_valid[PARALLELISM-1:0];
  logic mult_out_ready[PARALLELISM-1:0];

  logic [OUT_WIDTH-1:0] mult_cast_data[PARALLELISM-1:0];

  logic [OUT_WIDTH-1:0] out_reg_data[PARALLELISM-1:0];
  logic out_reg_valid[PARALLELISM-1:0];
  logic out_reg_ready;


  // -----
  // Modules
  // -----

  split_n #(
      .N(3)
  ) input_split (
      .data_in_valid (in_valid),
      .data_in_ready (in_ready),
      .data_out_valid({local_max_in_valid, global_max_in_valid, local_values_in_valid}),
      .data_out_ready({local_max_in_ready, global_max_in_ready, local_values_in_ready})
  );

  fifo #(
      .DATA_WIDTH(IN_MAX_WIDTH),
      .DEPTH(4 * DEPTH)  // TODO: resize +1 ?
  ) local_max_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(in_max),
      .in_valid(local_max_in_valid),
      .in_ready(local_max_in_ready),
      .out_data(local_max_out),
      .out_valid(local_max_out_valid),
      .out_ready(local_max_out_ready),
      .empty(),
      .full()
  );

  comparator_accumulator #(
      .DATA_WIDTH(IN_MAX_WIDTH),
      .DEPTH(DEPTH),
      .MAX1_MIN0(1),
      .SIGNED(1)
  ) global_max_accumulator (
      .clk(clk),
      .rst(rst),
      .in_data(in_max),
      .in_valid(global_max_in_valid),
      .in_ready(global_max_in_ready),
      .out_data(global_max_out),
      .out_valid(global_max_out_valid),
      .out_ready(global_max_out_ready)
  );

  single_element_repeat #(
      .DATA_WIDTH(IN_MAX_WIDTH),
      .REPEAT(DEPTH)
  ) global_max_repeater (
      .clk(clk),
      .rst(rst),
      .in_data(global_max_out),
      .in_valid(global_max_out_valid),
      .in_ready(global_max_out_ready),
      .out_data(repeat_global_max_out),
      .out_valid(repeat_global_max_out_valid),
      .out_ready(repeat_global_max_out_ready)
  );

  matrix_fifo #(
      .DATA_WIDTH(IN_VALUE_WIDTH),
      .DIM0(PARALLELISM),
      .DIM1(1),
      .FIFO_SIZE(4 * DEPTH)  // TODO: resize?
  ) local_values_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(in_values),
      .in_valid(local_values_in_valid),
      .in_ready(local_values_in_ready),
      .out_data(local_values_out),
      .out_valid(local_values_out_valid),
      .out_ready(local_values_out_ready)
  );


  join2 subtract_join (
      .data_in_valid ({repeat_global_max_out_valid, local_max_out_valid}),
      .data_in_ready ({repeat_global_max_out_ready, local_max_out_ready}),
      .data_out_valid(subtract_in_valid),
      .data_out_ready(subtract_in_ready)
  );

  assign subtract_in_data = $signed(repeat_global_max_out) - $signed(local_max_out);

  skid_buffer #(
      .DATA_WIDTH(SUBTRACT_WIDTH)
  ) sub_reg (
      .clk(clk),
      .rst(rst),
      .data_in(subtract_in_data),
      .data_in_valid(subtract_in_valid),
      .data_in_ready(subtract_in_ready),
      .data_out(subtract_out_data),
      .data_out_valid(subtract_out_valid),
      .data_out_ready(subtract_out_ready)
  );

  join2 shift_join (
      .data_in_valid ({local_values_out_valid, subtract_out_valid}),
      .data_in_ready ({local_values_out_ready, subtract_out_ready}),
      .data_out_valid(shift_in_valid),
      .data_out_ready(shift_in_ready[0])
  );


  // Batched shift
  for (genvar i = 0; i < PARALLELISM; i++) begin : shift
    assign shift_in_data[i] = local_values_out[i] >> subtract_out_data;

    skid_buffer #(
        .DATA_WIDTH(IN_VALUE_WIDTH)
    ) shift_reg (
        .clk(clk),
        .rst(rst),
        .data_in(shift_in_data[i]),
        .data_in_valid(shift_in_valid),
        .data_in_ready(shift_in_ready[i]),
        .data_out(shift_out_data[i]),
        .data_out_valid(shift_out_valid[i]),
        .data_out_ready(shift_out_ready)
    );
  end

  split2 norm_split (
      .data_in_valid (shift_out_valid[0]),
      .data_in_ready (shift_out_ready),
      .data_out_valid({adjusted_values_in_valid, adder_tree_in_valid}),
      .data_out_ready({adjusted_values_in_ready, adder_tree_in_ready})
  );

  assign adjusted_values_in_data = shift_out_data;
  for (genvar i = 0; i < PARALLELISM; i++) begin : unsigned_hack
    assign adder_tree_in_data[i] = {1'b0, shift_out_data[i]};
  end

  matrix_fifo #(
      .DATA_WIDTH(IN_VALUE_WIDTH),
      .DIM0(PARALLELISM),
      .DIM1(1),
      .FIFO_SIZE(4 * DEPTH)  // TODO: resize?
  ) adjusted_values_buffer (
      .clk(clk),
      .rst(rst),
      .in_data(adjusted_values_in_data),
      .in_valid(adjusted_values_in_valid),
      .in_ready(adjusted_values_in_ready),
      .out_data(adjusted_values_out_data),
      .out_valid(adjusted_values_out_valid),
      .out_ready(adjusted_values_out_ready)
  );


  generate
    if (PARALLELISM == 1) begin : gen_skip_adder_tree
      assign adder_tree_out_data  = adder_tree_in_data[0];
      assign adder_tree_out_valid = adder_tree_in_valid;
      assign adder_tree_in_ready  = adder_tree_out_ready;
    end else begin : gen_adder_tree
      fixed_adder_tree #(
          .IN_SIZE (PARALLELISM),
          .IN_WIDTH(ADDER_TREE_IN_WIDTH)
      ) adder_tree (
          .clk(clk),
          .rst(rst),
          .data_in(adder_tree_in_data),
          .data_in_valid(adder_tree_in_valid),
          .data_in_ready(adder_tree_in_ready),
          .data_out(adder_tree_out_data),
          .data_out_valid(adder_tree_out_valid),
          .data_out_ready(adder_tree_out_ready)
      );
    end
  endgenerate

  fixed_accumulator #(
      .IN_DEPTH(DEPTH),
      .IN_WIDTH(ADDER_TREE_OUT_WIDTH)
  ) norm_accumulator (
      .clk(clk),
      .rst(rst),
      .data_in(adder_tree_out_data),
      .data_in_valid(adder_tree_out_valid),
      .data_in_ready(adder_tree_out_ready),
      .data_out(acc_out_data),
      .data_out_valid(acc_out_valid),
      .data_out_ready(acc_out_ready)
  );

  softermax_lpw_reciprocal #(
      .ENTRIES(8),
      .IN_WIDTH(ACC_WIDTH),
      .IN_FRAC_WIDTH(ACC_FRAC_WIDTH),
      .OUT_WIDTH(RECIP_WIDTH),
      .OUT_FRAC_WIDTH(RECIP_FRAC_WIDTH)
  ) norm_recip (
      .clk(clk),
      .rst(rst),
      .in_data(acc_out_data),
      .in_valid(acc_out_valid),
      .in_ready(acc_out_ready),
      .out_data(norm_recip_data),
      .out_valid(norm_recip_valid),
      .out_ready(norm_recip_ready)
  );

  single_element_repeat #(
      .DATA_WIDTH(RECIP_WIDTH),
      .REPEAT(DEPTH)
  ) repeat_norm_recip (
      .clk(clk),
      .rst(rst),
      .in_data(norm_recip_data),
      .in_valid(norm_recip_valid),
      .in_ready(norm_recip_ready),
      .out_data(repeat_norm_recip_data),
      .out_valid(repeat_norm_recip_valid),
      .out_ready(repeat_norm_recip_ready)
  );

  join2 mult_join (
      .data_in_valid ({adjusted_values_out_valid, repeat_norm_recip_valid}),
      .data_in_ready ({adjusted_values_out_ready, repeat_norm_recip_ready}),
      .data_out_valid(mult_in_valid),
      .data_out_ready(mult_in_ready[0])
  );

  // Batched mult & cast
  for (genvar i = 0; i < PARALLELISM; i++) begin : output_mult_cast
    assign mult_in_data[i] = adjusted_values_out_data[i] * repeat_norm_recip_data;

    skid_buffer #(
        .DATA_WIDTH(MULT_WIDTH)
    ) mult_reg (
        .clk(clk),
        .rst(rst),
        .data_in(mult_in_data[i]),
        .data_in_valid(mult_in_valid),
        .data_in_ready(mult_in_ready[i]),
        .data_out(mult_out_data[i]),
        .data_out_valid(mult_out_valid[i]),
        .data_out_ready(mult_out_ready[0])
    );

    fixed_signed_cast #(
        .IN_WIDTH(MULT_WIDTH),
        .IN_FRAC_WIDTH(MULT_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data (mult_out_data[i]),
        .out_data(mult_cast_data[i])
    );

    skid_buffer #(
        .DATA_WIDTH(MULT_WIDTH)
    ) out_reg (
        .clk(clk),
        .rst(rst),
        .data_in(mult_cast_data[i]),
        .data_in_valid(mult_out_valid[i]),
        .data_in_ready(mult_out_ready[i]),
        .data_out(out_reg_data[i]),
        .data_out_valid(out_reg_valid[i]),
        .data_out_ready(out_reg_ready)
    );
  end

  assign out_data = out_reg_data;
  assign out_valid = out_reg_valid[0];
  assign out_reg_ready = out_ready;

endmodule
