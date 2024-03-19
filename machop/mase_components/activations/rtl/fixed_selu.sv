`timescale 1ns / 1ps
module fixed_selu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 32,
    parameter DATA_IN_0_PRECISION_1 = 16,
	parameter DATA_IN_0_PRECISION_INT = DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1-1, //number of integer bits
	
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 32,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 16,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0,
	
	parameter SCALE_PRECISION_1 = 16,	//frac width, max 32
	parameter ALPHA_PRECISION_1 = 16,	//frac width, max 32

    parameter INPLACE = 0
) (
	/* verilator lint_off UNUSEDSIGNAL */
	/* verilator lint_off SELRANGE */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);
	logic signed [SCALE_PRECISION_1+1:0] scale_fixed;
	logic signed [ALPHA_PRECISION_1+1:0] alpha_fixed;
	
	const logic signed [33:0] alpha = 34'b0110101100010110101111101011010111;
	const logic signed [33:0] scale = 34'b0100001100111110101011110101101010;
	
	localparam L1= 16+16+DATA_OUT_0_PRECISION_1+DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_1-1;
	localparam L2= 16+16+DATA_OUT_0_PRECISION_1-16-DATA_IN_0_PRECISION_1;
	
	fixed_round #(
		.IN_WIDTH(34),            // Set the parameter values
		.IN_FRAC_WIDTH(32),
		.OUT_WIDTH(SCALE_PRECISION_1+2),
		.OUT_FRAC_WIDTH(SCALE_PRECISION_1)
	) fixed_round_inst (
		.data_in(scale),       // Connect inputs and outputs
		.data_out(scale_fixed)
	);
	
	fixed_round #(
		.IN_WIDTH(34),            // Set the parameter values
		.IN_FRAC_WIDTH(32),
		.OUT_WIDTH(ALPHA_PRECISION_1+2),
		.OUT_FRAC_WIDTH(ALPHA_PRECISION_1)
	) fixed_round_inst2 (
		.data_in(alpha),       // Connect inputs and outputs
		.data_out(alpha_fixed)
	);	

    logic [DATA_OUT_0_PRECISION_0-1:0] exp_out[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];

    for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1; i++) begin : SeLU
        logic signed [DATA_IN_0_PRECISION_0-1:0] signed_data_in;
        logic [DATA_IN_0_PRECISION_0-1:0] abs_data_in;
        logic signed [DATA_OUT_0_PRECISION_0-1:0] value_1;
        logic signed [DATA_OUT_0_PRECISION_0:0] sub_value;   
        logic signed [DATA_OUT_0_PRECISION_0+1+ALPHA_PRECISION_1+2-1:0] alpha_pdt;   
        logic signed [DATA_OUT_0_PRECISION_0+1+ALPHA_PRECISION_1+2+SCALE_PRECISION_1+2-1:0] scale_pdt;    
    
        assign signed_data_in = $signed(data_in_0[i]);
        assign abs_data_in    = (signed_data_in <= 0) ? $unsigned(-signed_data_in) : signed_data_in;
    
         exp #(
            .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
            .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
            .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
            .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
         ) exp_inst (
            .rst(rst),
            .clk(clk),
            .data_in_0(abs_data_in),
            .data_out_0(exp_out[i])
         );
         
        always_comb begin 
             
            // Scale and add alpha
            if (signed_data_in <= 0) begin
                value_1 = 0;
                value_1[DATA_OUT_0_PRECISION_1]= 1;
                sub_value = (exp_out[i] - value_1);
                alpha_pdt = alpha_fixed * sub_value;
                scale_pdt = scale_fixed * alpha_pdt;
            end else begin
				value_1 = 0;
                sub_value = 0;
                alpha_pdt = 0;		
                scale_pdt = 0;
                scale_pdt[L1:L2] =  scale_fixed *signed_data_in;
            end
        end
        
        fixed_round #(
            .IN_WIDTH(DATA_OUT_0_PRECISION_0+1+ALPHA_PRECISION_1+2+SCALE_PRECISION_1+2),            // Set the parameter values
            .IN_FRAC_WIDTH(DATA_OUT_0_PRECISION_1+ALPHA_PRECISION_1+SCALE_PRECISION_1),
            .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
            .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
        ) fixed_round_inst2 (
            .data_in(scale_pdt),       // Connect inputs and outputs
            .data_out(data_out_0[i])
        );     
    end

    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;
	
endmodule

