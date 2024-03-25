`timescale 1ns / 1ps

module top();


LLMint#(
    .ORIGINAL_PRECISION(32),
    .REDUCED_PRECISION(8),
    .TENSOR_SIZE_DIM(4),
    .WEIGHT_DIM_0(6),
    .WEIGHT_DIM_1(6),
    .HIGH_SLOTS(2),
    .THRESHOLD(6)
)llm_int (
    .data_in_valid(),
    .data_in_ready(),
    .weight_valid(),
    .weight_ready(),
    .data_out_ready(),
    .data_out_valid(),
    .data_in(),
    // We combine weights and quantized weights into a single array
    .weights(),
    .data_out()
);

endmodule