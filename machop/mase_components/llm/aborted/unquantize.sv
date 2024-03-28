`timescale 1ns / 1ps

module unquantize #(

    parameter IN_WIDTH = 8,  // FP16
    parameter IN_SIZE = 4,  // in cols
    parameter IN_PARALLELISM = 1, // in rows

    parameter OUT_WIDTH = 16, // TODO
    parameter OUT_ROWS = IN_PARALLELISM,
    parameter OUT_COLUMNS = IN_SIZE,

    parameter QUANTIZATION_WIDTH = IN_WIDTH,
    parameter MAX_NUM_WIDTH = OUT_WIDTH
) (
    input clk,
    input rst,

    // data_in
    input  [IN_WIDTH-1:0] data_in      [IN_PARALLELISM * IN_SIZE-1: 0],
    input data_in_valid,
    output data_in_ready,

    input [MAX_NUM_WIDTH-1:0] max_num, //TODO: change datatype

    // data_out
    output [OUT_WIDTH-1:0] data_out      [OUT_ROWS * OUT_COLUMNS - 1 :0],
    output data_out_valid,
    input data_out_ready    
);




localparam RESHAPE_FACTOR_FRAC_WIDITH = 16;
logic [IN_WIDTH + MAX_NUM_WIDTH-1:0] data_out_unrounded      [OUT_ROWS * OUT_COLUMNS - 1 :0];
logic [IN_WIDTH + MAX_NUM_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] data_out_unrounded_reshape      [OUT_ROWS * OUT_COLUMNS - 1 :0];
logic [QUANTIZATION_WIDTH-1:0] scale_constant;

// assign scale_constant = (1 << QUANTIZATION_WIDTH)-1;
assign scale_constant = 8'b01111111;


for (genvar i = 0; i < IN_SIZE * IN_PARALLELISM; i = i + 1) begin: QUANTIZE
    fixed_mult #(
    .IN_A_WIDTH(IN_WIDTH),
    .IN_B_WIDTH(MAX_NUM_WIDTH)
    ) fixed_mult_inst(
    .data_a(data_in[i]),
    .data_b(max_num),
    .product(data_out_unrounded[i])
    );


    logic [IN_WIDTH + MAX_NUM_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] data_out_unrounded_temp;
    logic [IN_WIDTH + MAX_NUM_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] temp1;
    logic [IN_WIDTH + MAX_NUM_WIDTH+RESHAPE_FACTOR_FRAC_WIDITH-1:0] temp2;
    assign data_out_unrounded_temp = $signed(data_out_unrounded[i]);
    assign temp1 = data_out_unrounded_temp << RESHAPE_FACTOR_FRAC_WIDITH;
    assign temp2 = $signed($signed(temp1) / $signed(scale_constant));

    // assign data_out_unrounded_reshape[i] = (data_out_unrounded_temp<<RESHAPE_FACTOR_FRAC_WIDITH)/ max_num;
    assign data_out_unrounded_reshape[i] = temp2;
end

fixed_rounding #(
    .IN_SIZE(OUT_ROWS*OUT_COLUMNS),
    .IN_WIDTH(IN_WIDTH + MAX_NUM_WIDTH + RESHAPE_FACTOR_FRAC_WIDITH),
    .IN_FRAC_WIDTH(RESHAPE_FACTOR_FRAC_WIDITH),
    .OUT_WIDTH(OUT_WIDTH),
    .OUT_FRAC_WIDTH(0)
) fixed_rounding_inst(
    .data_in(data_out_unrounded_reshape), 
    .data_out(data_out) 
);

endmodule







// logic [IN_WIDTH-1:0] scale_factor; //TODO: change datatype

// localparam SCALE_FACTOR_FRAC_WIDITH = 8;
// localparam SCALE_FACTOR_WIDTH = 16;

// assign scale_factor = max_mum >> IN_WIDTH; //TODO: VERY important.





