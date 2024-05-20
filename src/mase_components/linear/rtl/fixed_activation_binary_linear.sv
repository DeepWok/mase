`timescale 1ns / 1ps
module fixed_activation_binary_linear #(
    parameter IN_WIDTH = 32,
    parameter IN_FRAC_WIDTH = 0,
    parameter IN_SIZE = 4,
    parameter IN_DEPTH = 3,
    parameter PARALLELISM = 2,

    // DO NOT MODIFY: parameterised for binary weights
    parameter WEIGHT_WIDTH = 1,
    parameter WEIGHT_FRAC_WIDTH = 0,
    parameter WEIGHT_SIZE = IN_SIZE * PARALLELISM,

    // This is the width for the summed product
    // +1 is because of the bias
    parameter HAS_BIAS = 0,
    parameter OUT_WIDTH = IN_WIDTH + $clog2(IN_SIZE) + $clog2(IN_DEPTH) + HAS_BIAS,
    /* verilator lint_off UNUSEDPARAM */
    parameter OUT_FRAC_WIDTH = IN_FRAC_WIDTH + WEIGHT_FRAC_WIDTH,
    /* verilator lint_on UNUSEDPARAM */
    parameter OUT_SIZE = PARALLELISM,

    parameter BIAS_SIZE = OUT_SIZE,
    parameter BIAS_WIDTH = 32,
    /* verilator lint_off UNUSEDPARAM */
    parameter BIAS_FRAC_WIDTH = 0
    /* verilator lint_on UNUSEDPARAM */
) (
    input clk,
    input rst,

    // input port for data_inivations
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE-1:0],
    input                 data_in_valid,
    output                data_in_ready,

    // input port for weight
    input  [WEIGHT_WIDTH-1:0] weight      [WEIGHT_SIZE-1:0],
    input                     weight_valid,
    output                    weight_ready,

    /* verilator lint_off UNUSEDSIGNAL */
    input  [BIAS_WIDTH-1:0] bias          [BIAS_SIZE-1:0],
    input                   bias_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output                  bias_ready,
    output [ OUT_WIDTH-1:0] data_out      [ OUT_SIZE-1:0],
    output                  data_out_valid,
    input                   data_out_ready
);

  // fixed activation binary dot product
  localparam fabdp_WIDTH = IN_WIDTH + $clog2(IN_SIZE);
  localparam ACC_WIDTH = fabdp_WIDTH + $clog2(IN_DEPTH);

  logic fabdp_join_valid, fabdp_join_ready;
  join2 #() fabdp_join_inst (
      .data_in_ready ({weight_ready, data_in_ready}),
      .data_in_valid ({weight_valid, data_in_valid}),
      .data_out_valid(fabdp_join_valid),
      .data_out_ready(fabdp_join_ready)
  );

  /* verilator lint_off UNUSEDSIGNAL */
  // Assume the parallelised hardware above have the same arrival time
  // which means that they always have the same state. So we can just
  // pick one of the valid signal to use.
  logic [PARALLELISM-1:0] fabdp_data_ready, fabdp_weight_ready;
  assign fabdp_join_ready = fabdp_data_ready[0];
  /* verilator lint_on UNUSEDSIGNAL */

  logic                 acc_ready;
  logic [ACC_WIDTH-1:0] acc_data_out[PARALLELISM-1:0];

  // There are PARALLELISM number of dot product instances with IN_SIZE inputs 
  // and each one computes for IN_DEPTH iterations for each inputs.
  for (genvar i = 0; i < PARALLELISM; i = i + 1) begin : linear
    // Assume the weight are transposed and partitioned 
    logic [WEIGHT_WIDTH-1:0] current_weight[IN_SIZE-1:0];
    assign current_weight = weight[IN_SIZE*i+IN_SIZE-1:IN_SIZE*i];

    logic [fabdp_WIDTH-1:0] fabdp_data_out;
    logic fabdp_data_out_valid, fabdp_data_out_ready;

    // The inputs are already sync-ed by the previous join
    fixed_activation_binary_dot_product #(
        .IN_WIDTH(IN_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .IN_SIZE(IN_SIZE)
    ) fabdp_inst (
        .clk(clk),
        .rst(rst),
        .data_in(data_in),
        .data_in_valid(fabdp_join_valid),
        .data_in_ready(fabdp_data_ready[i]),
        .weight(current_weight),
        .weight_valid(fabdp_join_valid),
        .weight_ready(fabdp_weight_ready[i]),
        .data_out(fabdp_data_out),
        .data_out_valid(fabdp_data_out_valid),
        .data_out_ready(fabdp_data_out_ready)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic acc_data_out_valid, acc_data_out_ready;
    /* verilator lint_on UNUSEDSIGNAL */

    fixed_accumulator #(
        .IN_WIDTH(fabdp_WIDTH),
        .IN_DEPTH(IN_DEPTH)
    ) fixed_accumulator_inst (
        .clk(clk),
        .rst(rst),
        .data_in(fabdp_data_out),
        .data_in_valid(fabdp_data_out_valid),
        .data_in_ready(fabdp_data_out_ready),
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

    logic acc_join_valid, acc_join_ready;
    join2 #() acc_join_inst (
        .data_in_ready ({bias_ready, acc_ready}),
        .data_in_valid ({bias_valid, linear[0].acc_data_out_valid}),
        .data_out_valid(acc_join_valid),
        .data_out_ready(acc_join_ready)
    );
    logic [PARALLELISM-1:0] reg_ready;
    assign acc_join_ready = &reg_ready;

    logic [ACC_WIDTH-1:0] bias_sext[PARALLELISM-1:0];
    fixed_cast #(
        .IN_SIZE(PARALLELISM),
        .IN_WIDTH(BIAS_WIDTH),
        .IN_FRAC_WIDTH(BIAS_FRAC_WIDTH),
        .OUT_WIDTH(ACC_WIDTH),
        .OUT_FRAC_WIDTH(IN_FRAC_WIDTH + WEIGHT_FRAC_WIDTH)
    ) bias_cast (
        .data_in (bias),
        .data_out(bias_sext)
    );

    for (genvar i = 0; i < PARALLELISM; i = i + 1) begin : add_bias
      logic [OUT_WIDTH-1:0] add;
      assign add = $signed(acc_data_out[i]) + $signed(bias_sext[i]);
      /* verilator lint_off UNUSEDSIGNAL */
      logic dout_valid;
      register_slice #(
          .IN_WIDTH(OUT_WIDTH)
      ) register_slice (
          .clk           (clk),
          .rst           (rst),
          .data_in_valid (acc_join_valid),
          .data_in_ready (reg_ready[i]),
          .data_in_data  (add),
          .data_out_valid(dout_valid),
          .data_out_ready(data_out_ready),
          .data_out_data (data_out[i])
      );
    end
    assign data_out_valid = add_bias[0].dout_valid;

  end else begin
    assign acc_ready = data_out_ready;
    assign data_out_valid = linear[0].acc_data_out_valid;
    assign data_out = acc_data_out;
    assign bias_ready = 1;
  end

endmodule
