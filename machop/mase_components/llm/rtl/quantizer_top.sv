`timescale 1ns / 1ps

module quantizer_top #(
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 1, // in rows

    parameter OUT_WIDTH = 8, // int8
    parameter OUT_ROWS = IN_PARALLELISM,
    parameter OUT_COLUMNS = IN_SIZE,

    parameter QUANTIZATION_WIDTH = OUT_WIDTH,

    parameter MAX_NUM_WIDTH = IN_WIDTH
) (
    input clk,
    input rst,

    // data_in
    input  [IN_WIDTH-1:0] data_in      [IN_PARALLELISM * IN_SIZE-1: 0],
    input data_in_valid,
    output data_in_ready,

    // data_out
    output [OUT_WIDTH-1:0] data_out      [OUT_ROWS * OUT_COLUMNS - 1 :0],

    output logic data_out_valid,
    input data_out_ready,   

    output [MAX_NUM_WIDTH-1:0] max_num  

);


logic [IN_WIDTH-1:0] data_in_buffered      [IN_PARALLELISM * IN_SIZE-1: 0];
logic [MAX_NUM_WIDTH-1:0] max_reg;

find_max #(
    .IN_WIDTH(IN_WIDTH),  // FP16
    .IN_SIZE(IN_SIZE),  // in cols
    .IN_PARALLELISM(IN_PARALLELISM), // in rows
    .MAX_NUM_WIDTH (MAX_NUM_WIDTH)
) find_max_inst(
    .clk(clk),
    .rst(rst),

    // data_in
    .data_in (data_in),
    .data_in_valid(data_in_valid),
    .data_in_ready(data_in_ready),

    // data_out
    .data_out(data_in_buffered),

    .data_out_valid(data_out_valid),
    .data_out_ready(data_out_ready),   

    // Assume max_num is synchronous with data_out
    // so valid and ready signal for max_num are not needed here
    .max_num(max_reg)  //TODO: change datatype
);

quantizer_part #(
    .IN_WIDTH(IN_WIDTH),  // FP16
    .IN_SIZE(IN_SIZE),  // in cols
    .IN_PARALLELISM(IN_PARALLELISM),// in rows
    .OUT_WIDTH(OUT_WIDTH), // int8
    .MAX_NUM_WIDTH (MAX_NUM_WIDTH)
) quantizer_part_inst(
    .clk(clk),
    .rst(rst),
    .data_in(data_in_buffered),
    .max_num_in(max_reg),
    .data_out(data_out),
    .max_num_out(max_num)  //TODO: change datatype
);

endmodule