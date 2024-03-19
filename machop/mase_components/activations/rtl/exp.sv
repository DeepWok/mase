`timescale 1ns / 1ps
module exp #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 32,
    parameter DATA_IN_0_PRECISION_1 = 16,
	parameter DATA_IN_0_PRECISION_INT = DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1, //number of integer bits
	
    parameter DATA_OUT_0_PRECISION_0 = 32,
    parameter DATA_OUT_0_PRECISION_1 = 16,
	
	parameter LUT_1_PRECISION_0 = 17,
    parameter LUT_1_PRECISION_1 = 16,
	
	parameter LUT_2_PRECISION_0 = 17,
    parameter LUT_2_PRECISION_1 = 16,
	
	parameter IMPRECISE_PRECISION_0 = LUT_1_PRECISION_0
	//parameter IMPRECISE_PRECISION_1= 3*DATA_IN_0_PRECISION_1+5,
	
	//parameter MUL_PRECISION_0 = 8,
    //parameter MUL_PRECISION_1 = 0
	
) (
    /* verilator lint_off UNUSEDSIGNAL */
	 /* verilator lint_off SELRANGE */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0,
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0
);
    localparam PDT_WIDTH = LUT_1_PRECISION_0+LUT_2_PRECISION_0+IMPRECISE_PRECISION_0;
    logic [DATA_IN_0_PRECISION_INT-1:0] a_sat;
	logic [3:0] a_precise_1;
	logic [2:0] a_precise_2;
	logic [DATA_IN_0_PRECISION_1-1:0] a_imprecise;
	logic [LUT_1_PRECISION_0-1:0] exp_precise_1;
	logic [LUT_2_PRECISION_0-1:0] exp_precise_2;
	logic [IMPRECISE_PRECISION_0-1:0] exp_imprecise;
	logic [PDT_WIDTH-1:0] product;
	
	always_comb begin
        if (DATA_IN_0_PRECISION_INT > 4) //there is possibility for saturation
		begin		  
			a_sat[DATA_IN_0_PRECISION_INT-4-1:0] = data_in_0[DATA_IN_0_PRECISION_0-1:DATA_IN_0_PRECISION_1+4];
			a_sat[DATA_IN_0_PRECISION_INT-1:DATA_IN_0_PRECISION_INT-4] = 0;
			if (a_sat > 0) 
			begin
				a_precise_1 = 4'b1111;
				if (DATA_IN_0_PRECISION_1 > 3)
				begin
					a_precise_2 = 3'b111;
					a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = {(DATA_IN_0_PRECISION_1-3){1'b1}};
					a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
				end
				else if (DATA_IN_0_PRECISION_1 == 3)
				begin
					a_precise_2 = {(DATA_IN_0_PRECISION_1){1'b1}};
					a_imprecise = 0;
				end
				else if (DATA_IN_0_PRECISION_1 == 0)
				begin
					a_precise_2 = 0;
					a_imprecise = 0;
				end
				else
				begin
					a_precise_2[2:2-DATA_IN_0_PRECISION_1+1]={(DATA_IN_0_PRECISION_1){1'b1}};
					a_precise_2[2-DATA_IN_0_PRECISION_1:0]= 0;
					a_imprecise = 0;
				end 
			end
			else
			begin 
				a_precise_1 = data_in_0[DATA_IN_0_PRECISION_1+4-1:DATA_IN_0_PRECISION_1];
				if (DATA_IN_0_PRECISION_1 > 3)
				begin
					a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
					a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
					a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
				end
				else if (DATA_IN_0_PRECISION_1 == 3)
				begin
					a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
					a_imprecise = 0;
				end
				else if (DATA_IN_0_PRECISION_1 == 0)
				begin
					a_precise_2 = 0;
					a_imprecise = 0;
				end
				else
				begin
					a_precise_2[2:2-DATA_IN_0_PRECISION_1+1]=data_in_0[DATA_IN_0_PRECISION_1-1:0];
					a_precise_2[2-DATA_IN_0_PRECISION_1:0] = 0;
					a_imprecise = 0;
				end 
			end
		end	
		else if(DATA_IN_0_PRECISION_INT == 4)	
		begin
			a_precise_1 = data_in_0[DATA_IN_0_PRECISION_0-1:DATA_IN_0_PRECISION_1];
			if (DATA_IN_0_PRECISION_1 > 3)
			begin
				a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
				a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
				a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
			end
			else if (DATA_IN_0_PRECISION_1 == 3)
			begin
				a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
				a_imprecise = 0;
			end
			else if (DATA_IN_0_PRECISION_1 == 0)
			begin
				a_precise_2 = 0;
				a_imprecise = 0;
			end
			else
			begin
				a_precise_2[2:2-DATA_IN_0_PRECISION_1+1]=data_in_0[DATA_IN_0_PRECISION_1-1:0];
				a_precise_2[2-DATA_IN_0_PRECISION_1:0]  = 0;
				a_imprecise = 0;
			end 			
		end	
		else if(DATA_IN_0_PRECISION_INT == 0)	
		begin
			a_precise_1 = 0;
			if (DATA_IN_0_PRECISION_1 > 3)
			begin
				a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
				a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
				a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
			end
			else if (DATA_IN_0_PRECISION_1 == 3)
			begin
				a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
				a_imprecise = 0;
			end
			else if (DATA_IN_0_PRECISION_1 == 0)
			begin
				a_precise_2 = 0;
				a_imprecise = 0;
			end
			else
			begin
				a_precise_2[2:2-DATA_IN_0_PRECISION_1+1]=data_in_0[DATA_IN_0_PRECISION_1-1:0];
				a_precise_2[2-DATA_IN_0_PRECISION_1:0]  = 0;
				a_imprecise = 0;
			end 			
		end
		else
		begin
			a_precise_1[DATA_IN_0_PRECISION_INT-1:0]=data_in_0[DATA_IN_0_PRECISION_0-1:DATA_IN_0_PRECISION_1];
			a_precise_1[3:DATA_IN_0_PRECISION_INT] = 0;
			if (DATA_IN_0_PRECISION_1 > 3)
			begin
				a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
				a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
				a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
			end
			else if (DATA_IN_0_PRECISION_1 == 3)
			begin
				a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
				a_imprecise = 0;
			end
			else if (DATA_IN_0_PRECISION_1 == 0)
			begin
				a_precise_2 = 0;
				a_imprecise = 0;
			end
			else
			begin
				a_precise_2[2:2-DATA_IN_0_PRECISION_1+1]=data_in_0[DATA_IN_0_PRECISION_1-1:0];
				a_precise_2[2-DATA_IN_0_PRECISION_1:0]  = 0;
				a_imprecise = 0;
			end 		
		end 
    end
    
    integer_lut_16 integer_lut_inst
    (
        .address(a_precise_1),
        .data_out(exp_precise_1)
    );
    
    fractional_lut_16 fractional_lut_inst
    (
        .address(a_precise_2),
        .data_out(exp_precise_2)
    );

    series_approx #(
    .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_1),
    .DATA_OUT_0_PRECISION_0(IMPRECISE_PRECISION_0-1)
    ) series_approx_inst (
    .data_in_0(a_imprecise),
    .data_out_0(exp_imprecise[IMPRECISE_PRECISION_0-2:0])
    );
    assign exp_imprecise[IMPRECISE_PRECISION_0-1] = 0;
    
    assign product = exp_precise_1*exp_precise_2*exp_imprecise;
    
   // assign data_out_0 = product[PDT_WIDTH-2-1: PDT_WIDTH - DATA_OUT_0_PRECISION_0]; //always less than or equal to 1
    
    fixed_round #(
        .IN_WIDTH(PDT_WIDTH),            // Set the parameter values
        .IN_FRAC_WIDTH(PDT_WIDTH-3),
        .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
    ) fixed_round_inst1 (
        .data_in(product),       // Connect inputs and outputs
        .data_out(data_out_0)
    );
    
    
    
    
    
    
    
    
    
    
endmodule
