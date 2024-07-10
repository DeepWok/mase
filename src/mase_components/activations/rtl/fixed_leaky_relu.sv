`timescale 1ns / 1ps

module fixed_leaky_relu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 16, //ensure that the value is atleast DATA_IN_0_PRECISION_0 + DATA_OUT_0_PRECISION_1
    parameter DATA_OUT_0_PRECISION_1 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    parameter INPLACE = 0,

    parameter NEGATIVE_SLOPE_PRECISION_0 = 8, //since negative slope is normally less than 1, NEGATIVE_SLOPE_PRECISION_1 ahould have more bits
    parameter NEGATIVE_SLOPE_PRECISION_1 = 7,
    parameter NEGATIVE_SLOPE_VALUE = 1

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

  initial begin
    assert (DATA_IN_0_PRECISION_0 == DATA_OUT_0_PRECISION_0)
    else $error("Leaky ReLU: DATA_IN_0_PRECISION_0 must be equal to DATA_OUT_0_PRECISION_0");
    assert (DATA_IN_0_PRECISION_0 == NEGATIVE_SLOPE_PRECISION_0)
    else $error("Leaky ReLU: DATA_IN_0_PRECISION_0 must be equal to NEGATIVE_SLOPE_PRECISION_0");
    assert (DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1)
    else $error("Leaky ReLU: DATA_IN_0_PRECISION_1 must be equal to DATA_OUT_0_PRECISION_1");
    assert (DATA_IN_0_PRECISION_1 == NEGATIVE_SLOPE_PRECISION_1)
    else $error("Leaky ReLU: DATA_IN_0_PRECISION_1 must be equal to NEGATIVE_SLOPE_PRECISION_1");
  end



  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if ($signed(data_in_0[i]) < 0)
        // check vivado behaviour
        // The behavior of leaky relu may not exactly match the behavior of the torch leaky relu.
        // This is becase of the multiplication with the negative slope is floored in system verilog instead of rounded.

        data_out_0[i] = ($signed(
            NEGATIVE_SLOPE_VALUE
        ) * $signed(
            data_in_0[i]
        )) >>> NEGATIVE_SLOPE_PRECISION_1;
      else data_out_0[i] = data_in_0[i];
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;


endmodule
