`timescale 1ns / 1ps

module quantizer #(
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

    output data_out_valid,
    input data_out_ready,   

    output [MAX_NUM_WIDTH-1:0] max_num  //TODO: change datatype
);

// logic [IN_WIDTH-1:0] max_num;// TODO: if max_num is 0, just skip this module?


// localparam SCALE_FACTOR_FRAC_WIDITH = 16;
// localparam SCALE_FACTOR_WIDTH = 32;
// logic [SCALE_FACTOR_WIDTH-1:0] scale_factor;//TODO: change datatype
logic [MAX_NUM_WIDTH-1:0] reg_max_num;
logic [MAX_NUM_WIDTH-1:0] reg_max_num_abs;

assign reg_max_num_abs = ($signed(reg_max_num) > 0) ? reg_max_num : -reg_max_num;
assign max_num = reg_max_num_abs;

logic cmp_in_valid, cmp_in_ready;
logic cmp_out_valid, cmp_out_ready;
assign cmp_in_valid = data_in_valid;
assign cmp_out_ready = data_out_ready;



fixed_comparator_tree #(
    .IN_SIZE(IN_SIZE*IN_PARALLELISM),
    .IN_WIDTH(IN_WIDTH),
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
localparam CMP_TREE_DELAY = $clog2(IN_SIZE*IN_PARALLELISM) + 1;  // increased by 1 so the fifo is never full

for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: PARALLEL_FIFO
    logic fifo_in_valid, fifo_in_ready;
    logic fifo_out_valid, fifo_out_ready;
    logic fifo_empty;
    assign fifo_in_valid = data_in_valid;
    assign fifo_out_ready = data_out_ready;
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

// logic join2_valid, join2_ready;
// join2 #(
// ) (
//     input logic [1:0] data_in_valid,
//     output logic [1:0] data_in_ready,
//     .data_out_valid(join2_valid),
//     .data_out_ready(join2_ready)
// );

// TODO: assump that all fifos are operated synchronously
assign data_in_ready = (PARALLEL_FIFO[0].fifo_in_ready && !rst) && cmp_in_ready;
assign data_out_valid = PARALLEL_FIFO[0].fifo_out_valid && cmp_out_valid;

// // assign scale_factor = ((1 << (OUT_WIDTH-1))-1) << (SCALE_FACTOR_FRAC_WIDITH); //TODO?? reshape: make its width 2*IN_SIZE and do multiply first then divide
// logic [SCALE_FACTOR_WIDTH-1:0] temp;
// assign temp = (8'b01111111) << (SCALE_FACTOR_FRAC_WIDITH);  // temp = 127 << 8 = 0x7F00
// assign scale_factor = temp/reg_max_num_abs; //TODO?? reshape: make its width 2*IN_SIZE and do multiply first then divide

// assign scale_factor = max_num >> OUT_WIDTH; 

localparam RESHAPE_FACTOR_FRAC_WIDITH = 16;
logic [IN_WIDTH + QUANTIZATION_WIDTH-1:0] data_out_unrounded      [OUT_ROWS * OUT_COLUMNS - 1 :0];
logic [IN_WIDTH + QUANTIZATION_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] data_out_unrounded_reshape      [OUT_ROWS * OUT_COLUMNS - 1 :0];
logic [QUANTIZATION_WIDTH-1:0] scale_constant;

// assign scale_constant = (1 << QUANTIZATION_WIDTH)-1;
assign scale_constant = 8'b01111111;


for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: QUANTIZE
    fixed_mult #(
    .IN_A_WIDTH(IN_WIDTH),
    .IN_B_WIDTH(QUANTIZATION_WIDTH)
    ) fixed_mult_inst(
    .data_a(data_in_buffered[i]),
    .data_b(scale_constant),
    .product(data_out_unrounded[i])
    );



    logic [IN_WIDTH + QUANTIZATION_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] data_out_unrounded_temp;
    logic [IN_WIDTH + QUANTIZATION_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] temp1;
    logic [IN_WIDTH + QUANTIZATION_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] temp2;
    assign data_out_unrounded_temp = $signed(data_out_unrounded[i]);
    assign temp1 = data_out_unrounded_temp << RESHAPE_FACTOR_FRAC_WIDITH;
    assign temp2 = $signed($signed(temp1) / $signed(max_num));

    // assign data_out_unrounded_reshape[i] = (data_out_unrounded_temp<<RESHAPE_FACTOR_FRAC_WIDITH)/ max_num;
    assign data_out_unrounded_reshape[i] = temp2;
end

fixed_rounding #(
    .IN_SIZE(OUT_ROWS*OUT_COLUMNS),
    .IN_WIDTH(IN_WIDTH + QUANTIZATION_WIDTH + RESHAPE_FACTOR_FRAC_WIDITH),
    .IN_FRAC_WIDTH(RESHAPE_FACTOR_FRAC_WIDITH),
    .OUT_WIDTH(OUT_WIDTH),
    .OUT_FRAC_WIDTH(0)
) fixed_rounding_inst(
    .data_in(data_out_unrounded_reshape), 
    .data_out(data_out) 
);

endmodule








// logic [2*IN_WIDTH-1:0] data_quantize [IN_PARALLELISM * IN_SIZE-1: 0];//data type

// for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: MAX
// {
//     if (data_in[i] > max_num) begin
//     max_num = data_in[i];
//     end
//     else begin
//         max_num = max_num;
//     end
// }
// end


// data_quantize[i]=data_in[i] / max_num;
// data_out[i]=data_quantize[i] << OUT_WIDTH;