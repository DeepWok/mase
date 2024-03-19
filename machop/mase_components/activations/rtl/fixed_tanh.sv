`timescale 1ns / 1ps

module fixed_tanh #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 1,
	parameter DATA_IN_0_PRECISION_INT = DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1, //number of integer bits
	
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0
	
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);
	

	const logic signed [33 :0] a = 34'b0110000101000111101011100001010001; //2 integer
	logic signed [DATA_IN_0_PRECISION_0-1:0] a_fixed, b_fixed; //2 integer
	const logic signed [34 :0] b = 35'b01010010001111010111000010100011110; 
	
	 fixed_round #(
         .IN_WIDTH(34),            // Set the parameter values
         .IN_FRAC_WIDTH(32),
         .OUT_WIDTH(DATA_IN_0_PRECISION_0),
         .OUT_FRAC_WIDTH(DATA_IN_0_PRECISION_1)
     ) fixed_round_insta (
         .data_in(a),       // Connect inputs and outputs
         .data_out(a_fixed)
     );
	
	 fixed_round #(
         .IN_WIDTH(35),            // Set the parameter values
         .IN_FRAC_WIDTH(32),
         .OUT_WIDTH(DATA_IN_0_PRECISION_0),
         .OUT_FRAC_WIDTH(DATA_IN_0_PRECISION_1)
     ) fixed_round_instb (
         .data_in(b),       // Connect inputs and outputs
         .data_out(b_fixed)
     );	

	const logic signed [16 :0] m1 = 17'b11011101001110111;
	//const logic signed [17 :0] c1 = 17'b;
	const logic signed [16 :0] d1 = 17'b00000010000011000;
	const logic signed [16 :0] m2 = 17'b11110101001001100;
	const logic signed [16 :0] c2 = 17'b00110110100110001;
	const logic signed [16 :0] d2 = 17'b00111001110101111;

   
    for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1; i++) begin : tanh
           // Local variables for computation
        logic signed [DATA_IN_0_PRECISION_0-1:0] x_abs;      
        logic signed [DATA_IN_0_PRECISION_0-1:0] x_abs_dum;
        logic signed [2*DATA_IN_0_PRECISION_0-1:0] x_squared;
        logic signed [2*DATA_IN_0_PRECISION_0+17-1:0] temp_result;
        logic signed [DATA_IN_0_PRECISION_0+17-1:0] term0;
        logic signed [2*DATA_IN_0_PRECISION_0+17-1:0] term1;
        logic signed [2*DATA_IN_0_PRECISION_0+17-1:0] term2;
        logic signed [DATA_OUT_0_PRECISION_0-1:0] temp_out;

        assign x_abs = ($signed(data_in_0[i]) >= 0) ? data_in_0[i] : -data_in_0[i];

        assign x_abs_dum = x_abs;
        assign x_squared = x_abs * x_abs;

        always_comb begin 
             
            if (x_abs_dum <= a_fixed) begin
				term0 = 0;
                term1 = 0;  
                term1[2*DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+17-1 -1:DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+17-1] = x_abs;    
                //term1 = {x_abs[DATA_IN_0_PRECISION_0-1],x_abs[DATA_IN_0_PRECISION_0-2:0]<<(DATA_IN_0_PRECISION_1+16)};
                term2 = 0;  
                term2[16+2*DATA_IN_0_PRECISION_1-1:2*DATA_IN_0_PRECISION_1] = d1[15:0];        
                temp_result =  m1 * x_squared + term1 + term2;
            end
            else if (x_abs_dum <= b_fixed) begin              
                term0 = c2 * x_abs;
                
                term1 = 0;  
                term1[2*DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+16-1 -1:DATA_IN_0_PRECISION_0-DATA_IN_0_PRECISION_INT+1-1] = term0;    
                           
                term2 = 0;  
                term2[16+2*DATA_IN_0_PRECISION_1-1:2*DATA_IN_0_PRECISION_1] = d2[15:0];     
                temp_result = m2 * x_squared  + term1 + term2;
            end
            else begin
				term0 = 0; 
				term1 = 0; 
				term2 = 0;
                temp_result = 1<<(2* DATA_IN_0_PRECISION_1+16); // Output zero outside the specified range
            end
           
        end
        fixed_round #(
            .IN_WIDTH(2*DATA_IN_0_PRECISION_0+17),            // Set the parameter values
            .IN_FRAC_WIDTH(2*DATA_IN_0_PRECISION_1+16),
            .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
            .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
        ) fixed_round_inst (
            .data_in(temp_result),       // Connect inputs and outputs
            .data_out(temp_out)
        );
        
        assign data_out_0[i] = ($signed(data_in_0[i]) >= 0) ? temp_out: -temp_out;
  
    end

    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;
	
endmodule

