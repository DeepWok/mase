`timescale 1ns / 1ps
module fixed_relu #(
    parameter IN_WIDTH = 8,
    /* verilator lint_off UNUSEDPARAM */
    parameter IN_FRAC_WIDTH = 0,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 0,
    parameter OUT_SIZE = 0,
    /* verilator lint_on UNUSEDPARAM */
    parameter IN_SIZE = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    /* verilator lint_on UNUSEDSIGNAL */
    input logic [IN_WIDTH-1:0] data_in[IN_SIZE-1:0],
    output logic [IN_WIDTH-1:0] data_out[IN_SIZE-1:0],

    input  logic data_in_valid,
    output logic data_in_ready,
    output logic data_out_valid,
    input  logic data_out_ready
);

  for (genvar i = 0; i < IN_SIZE; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if ($signed(data_in[i]) <= 0) data_out[i] = '0;
      else data_out[i] = data_in[i];
    end
  end

  assign data_out_valid = data_in_valid;
  assign data_in_ready  = data_out_ready;

endmodule
