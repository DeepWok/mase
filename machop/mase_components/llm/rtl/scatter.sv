`timescale 1ns / 1ps
/*
 * ASSUMPTION: OUT_SMALL_COLUMNS == # zeros in ind_table
 */
module scatter #(
    /* verilator lint_off UNUSEDPARAM */
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_FRAC_WIDTH = 0,
    parameter IN_SIZE = 2,  // in cols
    parameter IN_PARALLELISM = 3,  // in rows
    parameter MAX_LARGE_NUMBERS = 3,
    parameter LARGE_NUMBER_THRES = 127  // a number larger than (BUT NOT EQUAL TO) THRES is counted as large number (outlier)
) (
    input clk,
    input rst,

    // input port for weight
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE * IN_PARALLELISM-1:0],
    input data_in_valid,
    output data_in_ready,

    output logic [IN_WIDTH-1:0] data_out_large [IN_SIZE * IN_PARALLELISM-1:0],
    output logic [IN_WIDTH-1:0] data_out_small [IN_SIZE * IN_PARALLELISM-1:0],
    output data_out_valid,
    input data_out_ready
);
    // Check if parameters are valid
    initial begin
        assert (IN_SIZE*IN_PARALLELISM > MAX_LARGE_NUMBERS) else $fatal("MAX_LARGE_NUMBERS must be less than input matrix size!");
        assert (LARGE_NUMBER_THRES > 0) else $fatal ("LARGE_NUMBER_THRES must be positive!");
    end

    logic [IN_SIZE * IN_PARALLELISM-1:0] outlier_flags;  
    // Checking all entries
    for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: MSB_CHECK
        fp16_comparator #(
            .IN_WIDTH (IN_WIDTH),
            .IN_FRAC_WIDTH (IN_FRAC_WIDTH),
            .THRES(LARGE_NUMBER_THRES)
        ) fp16_comp_inst(
            .data_in(data_in[i]),
            .result(outlier_flags[i])
        );
    end

    // Scatter entries until the number of outliers meets a threshold
    logic [IN_SIZE * IN_PARALLELISM-1:0] counter, inc; // counting all outliers of the input matrix
    always_comb begin: THRES_SCATTERING
        counter = 0;
        for (int ii = 0; ii < IN_SIZE * IN_PARALLELISM; ii = ii + 1) begin
            if ((outlier_flags[ii] == 1'b1) && (counter < MAX_LARGE_NUMBERS)) begin
                data_out_large[ii] = data_in[ii];
                data_out_small[ii] = 0;
                inc = 1;
            end else begin
                data_out_large[ii] = 0;
                data_out_small[ii] = data_in[ii];
                inc = 0;
            end
            counter = counter + inc;
        end
    end

    /* An alternativa but aborted stratey: process entries of one row in serial */
    // always_ff @(posedge clk) begin: entryAssignment
    //     if (ind_table[counter] == 1'b0) begin
    //         // small number, allocated to a small column
    //         current_data_out_small[counter] <= current_entry;
    //         current_data_out_large[counter] <= 0;
    //     end else begin
    //         // large number
    //         current_data_out_small[counter] <= 0;
    //         current_data_out_large[counter] <= current_entry;
    //     end
    // end

    assign data_out_valid = !rst && data_in_valid; //&& (ROW[0].counter == IN_SIZE);
    assign data_in_ready = !rst && data_out_ready;
endmodule
