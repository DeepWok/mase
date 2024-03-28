`timescale 1ns / 1ps

module find_max #(
    parameter IN_WIDTH = 16,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 1, // in rows

    parameter OUT_WIDTH = IN_WIDTH, // buffer_input FP16
    parameter OUT_ROWS = IN_PARALLELISM,
    parameter OUT_COLUMNS = IN_SIZE,
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

    output data_out_valid,
    input data_out_ready,   

    // Assume max_num is synchronous with data_out
    // so valid and ready signal for max_num are not needed here
    output [MAX_NUM_WIDTH-1:0] max_num  //TODO: change datatype
);

logic [MAX_NUM_WIDTH-1:0] reg_max_num;  // not absolute value!!

logic cmp_in_valid, cmp_in_ready;
logic cmp_out_valid, cmp_out_ready;



/* Broadcast valid control signals */
logic data_in_valid_buffer, data_in_ready_buffer;
// control signals for fifos
assign data_in_ready_buffer = PARALLEL_FIFO[0].fifo_in_ready;
for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: PARALLEL_VALID_CONTROL
    assign PARALLEL_FIFO[i].fifo_in_valid = data_in_valid_buffer;
end

split2 #(
) split2_inst(
    .data_in_valid(data_in_valid),
    .data_in_ready(data_in_ready),
    .data_out_valid({cmp_in_valid, data_in_valid_buffer}),
    .data_out_ready({cmp_in_ready, data_in_ready_buffer})
);


fixed_comparator_tree #(
    .IN_SIZE(IN_SIZE*IN_PARALLELISM),
    .IN_WIDTH(IN_WIDTH)
)fixed_comparator_tree_inst(
    .clk(clk),
    .rst(rst),
    /* verilator lint_on UNUSEDSIGNAL */
    .data_in(data_in),
    .data_in_valid(cmp_in_valid),
    .data_in_ready(cmp_in_ready),
    .data_out(reg_max_num),
    .data_out_valid(cmp_out_valid),
    .data_out_ready(cmp_out_ready)
);


logic [IN_WIDTH-1 :0] data_in_buffered [IN_SIZE*IN_PARALLELISM-1 :0]; 
localparam CMP_TREE_DELAY = $clog2(IN_SIZE*IN_PARALLELISM) + 10;  // increased by 10 so the fifo is never full

for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: PARALLEL_FIFO
    logic fifo_in_valid, fifo_in_ready;
    logic fifo_out_valid, fifo_out_ready;
    logic fifo_empty;
    // assign fifo_in_valid = data_in_valid;
    // assign fifo_out_ready = data_out_ready;
    // assign fifo_out_ready = cmp_out_valid;

    fifo #(
        .DEPTH (CMP_TREE_DELAY+1),
        .DATA_WIDTH (IN_WIDTH)
    ) data_in_fifo_inst (
        .clk (clk),
        .rst (rst),
        .in_data (data_in[i]),
        .in_valid (fifo_in_valid),
        .in_ready (fifo_in_ready),
        .out_data (data_in_buffered[i]),
        .out_valid (fifo_out_valid),
        .out_ready (fifo_out_ready),
        .empty (fifo_empty)
    );
end


// logic cmp_out_delay_valid, cmp_out_delay_ready;
// logic [MAX_NUM_WIDTH-1:0] reg_max_num_delay;

assign max_num = ($signed(reg_max_num) > 0) ? reg_max_num : -reg_max_num;

/* Broadcast ready control signals */
logic data_out_valid_buffer, data_out_ready_buffer;
join2 #(
) join2_inst(
    .data_in_valid({data_out_valid_buffer, cmp_out_valid}),
    .data_in_ready({data_out_ready_buffer, cmp_out_ready}),
    .data_out_valid(data_out_valid),
    .data_out_ready(data_out_ready)
);

assign data_out_valid_buffer = PARALLEL_FIFO[0].fifo_out_valid;
for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: PARALLEL_READY_CONTROL
    assign PARALLEL_FIFO[i].fifo_out_ready = data_out_ready_buffer;
end

assign data_out = data_in_buffered;


endmodule

