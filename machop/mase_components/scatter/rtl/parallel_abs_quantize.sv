`timescale 1ns / 1ps

module parallel_abs_quantize#(
    NO_INPUTS = 2,
    INPUT_PRECISION = 16,
    OUTPUT_PRECISION = 1
    )(
    input [INPUT_PRECISION-1:0] input_array [NO_INPUTS-1:0],
    output logic [OUTPUT_PRECISION-1:0] output_array [NO_INPUTS-1:0]
);

    logic sign  [NO_INPUTS-1:0];
    logic [OUTPUT_PRECISION-1:0] msbs [NO_INPUTS-1:0];

    for (genvar i = 0; i < NO_INPUTS; i = i + 1) begin
        //Extract sign bit
        assign sign[i] = input_array[i][INPUT_PRECISION-1];
        assign msbs[i] = input_array[i][INPUT_PRECISION-1:INPUT_PRECISION-1-OUTPUT_PRECISION];

        assign output_array[i] = (sign[i])? ~msbs[i]:msbs[i];
    end

endmodule