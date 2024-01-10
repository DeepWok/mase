module fixed_relu #(
    parameter IN_0_WIDTH = 8,
    parameter IN_0_SIZE = 8,
    /* verilator lint_off UNUSEDPARAM */
    parameter IN_0_FRAC_WIDTH = 0,

    parameter OUT_0_WIDTH = 8,
    parameter OUT_0_FRAC_WIDTH = 0,
    parameter OUT_0_SIZE = 0
    /* verilator lint_on UNUSEDPARAM */
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    /* verilator lint_on UNUSEDSIGNAL */
    input logic [IN_0_WIDTH-1:0] data_in_0[IN_0_SIZE-1:0],
    output logic [IN_0_WIDTH-1:0] data_out_0[IN_0_SIZE-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  for (genvar i = 0; i < IN_0_SIZE; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if ($signed(data_in_0[i]) <= 0) data_out_0[i] = '0;
      else data_out_0[i] = data_in_0[i];
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
