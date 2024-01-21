module fixed_relu #(
  /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0,

    parameter INPLACE = 0
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  for (genvar i = 0; i < DATA_IN_0_TENSOR_SIZE_DIM_0; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if ($signed(data_in_0[i]) <= 0) data_out_0[i] = '0;
      else data_out_0[i] = data_in_0[i];
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
