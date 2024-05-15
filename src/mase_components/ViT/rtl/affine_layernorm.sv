`timescale 1ns / 1ps
module affine_layernorm #(
    parameter IN_WIDTH = 32,
    parameter IN_FRAC_WIDTH = 0,
    parameter OUT_WIDTH = 6,
    parameter OUT_FRAC_WIDTH = 4,
    parameter BIAS_WIDTH = 8,
    parameter BIAS_FRAC_WIDTH = 4,
    parameter IN_SIZE = 4
) (
    input clk,
    input rst,

    // input port for data_inivations
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE-1:0],
    input                 data_in_valid,
    output                data_in_ready,

    // input port for weight
    input  [  IN_WIDTH-1:0] weight      [IN_SIZE-1:0],
    input  [BIAS_WIDTH-1:0] bias        [IN_SIZE-1:0],
    input                   weight_valid,
    input                   bias_valid,
    output                  weight_ready,
    output                  bias_ready,

    output [OUT_WIDTH-1:0] data_out      [IN_SIZE-1:0],
    output                 data_out_valid,
    input                  data_out_ready
);

  localparam PROD_WIDTH = IN_WIDTH + IN_WIDTH;
  localparam PROD_FRAC_WIDTH = IN_FRAC_WIDTH + IN_FRAC_WIDTH;
  logic [PROD_WIDTH - 1:0] prod[IN_SIZE - 1:0];
  logic pv_valid, pv_ready;
  logic [BIAS_WIDTH - 1:0] round_prod[IN_SIZE - 1:0];
  logic [BIAS_WIDTH:0] round_in[IN_SIZE - 1:0];
  logic wb_valid, wb_ready;

  fixed_vector_mult #(
      .IN_WIDTH(IN_WIDTH),
      .WEIGHT_WIDTH(IN_WIDTH),
      .IN_SIZE(IN_SIZE)
  ) fixed_vector_mult_inst (
      .clk(clk),
      .rst(rst),
      .data_in(data_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .weight(weight),
      .weight_valid(weight_valid),
      .weight_ready(weight_ready),
      .data_out(prod),
      .data_out_valid(pv_valid),
      .data_out_ready(pv_ready)
  );
  join2 #() join_inst2 (
      .data_in_ready ({pv_ready, bias_ready}),
      .data_in_valid ({pv_valid, bias_valid}),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );

  fixed_rounding #(
      .IN_SIZE(IN_SIZE),
      .IN_WIDTH(PROD_WIDTH),
      .IN_FRAC_WIDTH(PROD_FRAC_WIDTH),
      .OUT_WIDTH(BIAS_WIDTH),
      .OUT_FRAC_WIDTH(BIAS_FRAC_WIDTH)
  ) bias_cast (
      .data_in (prod),
      .data_out(round_prod)
  );
  for (genvar i = 0; i < IN_SIZE; i++) begin
    assign round_in[i] = {bias[i][BIAS_WIDTH-1], bias[i]} + {round_prod[i][BIAS_WIDTH-1], round_prod[i]};
  end

  fixed_rounding #(
      .IN_SIZE(IN_SIZE),
      .IN_WIDTH(BIAS_WIDTH + 1),
      .IN_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
  ) out_cast (
      .data_in (round_in),
      .data_out(data_out)
  );

endmodule
