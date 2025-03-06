module fixed_difflogic_logic #(
    parameter DATA_WIDTH,
    parameter NUM_NEURON,
    parameter [3:0] LAYER_OP_CODES [0:(NUM_NEURON-1)],
    parameter [$clog2(DATA_WIDTH)-1:0] IND_A [0:(NUM_NEURON-1)],
    parameter [$clog2(DATA_WIDTH)-1:0] IND_B [0:(NUM_NEURON-1)]
)(
    input wire clk,
    input wire rst,
    input wire [(DATA_WIDTH-1):0] data_in,
    output reg [(NUM_NEURON-1):0] data_out
);

genvar i;
generate

    for (i = 0; i < NUM_NEURON; i = i+1) begin: gen_block
        fixed_difflogic_logic_neuron #(
            .OP_CODE(LAYER_OP_CODES[i])
        ) neuron_inst (
            .clk(clk),
            .rst(rst),
            .a(data_in[IND_A[i]]),
            .b(data_in[IND_B[i]]),
            .res(data_out[i])
        );
    end

endgenerate

endmodule
