`timescale 1ns / 1ps

/*
 * Constrained by WEIGHT_PARALLELISM_DIM_0 == DATA_OUT_PARALLELISM_DIM_0
 *
*/

module mask #(
    /* verilator lint_off UNUSEDPARAM */
    parameter THRES = 30,

    // parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,
    parameter IN_PARALLELISM = 1
    // parameter OUT_COLUMNS = IN_SIZE
) (
    input clk,
    input rst,

    // input port for weight
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE * IN_PARALLELISM-1: 0],
    input data_in_valid,
    output data_in_ready,

    output [0:0] ind_table [IN_SIZE * IN_PARALLELISM-1:0],

    output [IN_WIDTH-1:0] data_out    [IN_SIZE * IN_PARALLELISM-1 :0],
    output data_out_valid,
    input data_out_ready
);

    // Large-number checking
    for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: MSB_CHECK
        fp16_comparator #(
            .THRES(THRES)
        ) fp16_comp_inst(
            .data_in(data_in[i]),
            .result(ind_table[i])
        );
    end

    assign data_out = data_in;
    // always_ff @(posedge clk) begin: RST
    //     if (rst) begin
    //         data_in_ready <= 0;
    //         data_out_valid <= 0;
    //     end else begin
    //         data_in_ready <= 1;
    //         data_out_valid <= 1;
    //     end
    // end
    assign data_in_ready = !rst && data_out_ready;
    assign data_out_valid = !rst && data_in_valid;
endmodule
