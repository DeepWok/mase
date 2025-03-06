module fixed_difflogic_logic #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter [3:0] LAYER_OP_CODES [0:(DATA_OUT_0_TENSOR_SIZE_DIM_0-1)],
    parameter [$clog2(DATA_IN_0_TENSOR_SIZE_DIM_0)-1:0] IND_A [0:(DATA_OUT_0_TENSOR_SIZE_DIM_0-1)],
    parameter [$clog2(DATA_IN_0_TENSOR_SIZE_DIM_0)-1:0] IND_B [0:(DATA_OUT_0_TENSOR_SIZE_DIM_0-1)]
)(
    input wire clk,
    input wire rst,

    input wire data_in_0_ready,
    input wire data_in_0_valid,
    input wire [(DATA_IN_0_TENSOR_SIZE_DIM_0-1):0] data_in_0,

    output wire data_out_0_ready,
    output wire data_out_0_valid,
    output reg [(DATA_OUT_0_TENSOR_SIZE_DIM_0-1):0] data_out_0
);

genvar i;
generate

    for (i = 0; i < DATA_OUT_0_TENSOR_SIZE_DIM_0; i = i+1) begin: gen_block
        fixed_difflogic_logic_neuron #(
            .OP_CODE(LAYER_OP_CODES[i])
        ) neuron_inst (
            .clk(clk),
            .rst(rst),
            .a(data_in_0[IND_A[i]]),
            .b(data_in_0[IND_B[i]]),
            .res(data_out_0[i])
        );
    end

endgenerate

endmodule
