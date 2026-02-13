`timescale 1ns / 1ps
module fixed_exp #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 32,  //total number of bits used to represent each input data
    parameter DATA_IN_0_PRECISION_1 = 16,  //fractional bits
    parameter DATA_IN_0_PRECISION_INT = DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1, //number of integer bits

    parameter DATA_OUT_0_PRECISION_0 = 32, //total number of bits used to represent each output data. Typically needs only (2 + output fractional) bits since range of negative exponential is from 0 to 1
    parameter DATA_OUT_0_PRECISION_1 = 16, //fractional bits. Output of the module is rounded to satisfy this value

    parameter LUT_1_PRECISION_0 = 17, //currently integer LUT stores 17-bit values with 16-bit precision. In the future, if another precision of LUT is used, change this parameter to corresponding width
    parameter LUT_1_PRECISION_1 = 16,

    parameter LUT_2_PRECISION_0 = 17, //currently integer LUT stores 17-bit values with 16-bit precision. In the future, if another precision of LUT is used, change this parameter to corresponding width
    parameter LUT_2_PRECISION_1 = 16,

    parameter IMPRECISE_PRECISION_0 = LUT_1_PRECISION_0 //number of bits used to represent output of series approximator
) (
    /* verilator lint_off UNUSEDSIGNAL */
    /* verilator lint_off SELRANGE */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0,
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0
);
  localparam PDT_WIDTH = LUT_1_PRECISION_0 + LUT_2_PRECISION_0 + IMPRECISE_PRECISION_0;
  logic [DATA_IN_0_PRECISION_INT-1:0] a_sat;  //store integer value greater than 15
  logic [3:0] a_precise_1;  //store integer value upto 15
  logic [2:0] a_precise_2;  //store most significant 3 fractional bits
  logic [DATA_IN_0_PRECISION_1-1:0] a_imprecise;  //store LSB fractional bits other than the MSB 3 bits
  logic [LUT_1_PRECISION_0-1:0] exp_precise_1;  //negative exponential of a_precise_1
  logic [LUT_2_PRECISION_0-1:0] exp_precise_2;  //negative exponential of a_precise_2
  logic [IMPRECISE_PRECISION_0-1:0] exp_imprecise;  //negative exponential of a_imprecise
  logic [PDT_WIDTH-1:0] product;  //final product of all part exponentials

  always_comb begin
    if (DATA_IN_0_PRECISION_INT > 4) begin  //there is possibility for saturation 
      a_sat[DATA_IN_0_PRECISION_INT-4-1:0] = data_in_0[DATA_IN_0_PRECISION_0-1:DATA_IN_0_PRECISION_1+4];
      a_sat[DATA_IN_0_PRECISION_INT-1:DATA_IN_0_PRECISION_INT-4] = 0;
      if (a_sat > 0) begin  //saturation condition. All bits of all parts are assigned to one.
        a_precise_1 = 4'b1111;
        if (DATA_IN_0_PRECISION_1 > 3) begin  //both a_precise_2 and a_imprecise are non-zero and assigned to ones				
          a_precise_2 = 3'b111;
          a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = {(DATA_IN_0_PRECISION_1 - 3) {1'b1}};
          a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
        end else if (DATA_IN_0_PRECISION_1 == 3) begin  //a_precise_2 is 3-bits of ones and a_imprecise is zero
          a_precise_2 = {(DATA_IN_0_PRECISION_1) {1'b1}};
          a_imprecise = 0;
        end else if (DATA_IN_0_PRECISION_1 == 0) begin  //both a_precise_2 and a_imprecise are zero
          a_precise_2 = 0;
          a_imprecise = 0;
        end else begin  //a_precise_2 is less than 3 bits  and a_imprecise are zero
          a_precise_2[2:2-DATA_IN_0_PRECISION_1+1] = {(DATA_IN_0_PRECISION_1) {1'b1}};
          a_precise_2[2-DATA_IN_0_PRECISION_1:0] = 0;
          a_imprecise = 0;
        end
      end else begin  //unsaturated condition
        a_precise_1 = data_in_0[DATA_IN_0_PRECISION_1+4-1:DATA_IN_0_PRECISION_1];
        if (DATA_IN_0_PRECISION_1 > 3) begin
          a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
          a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
          a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
        end else if (DATA_IN_0_PRECISION_1 == 3) begin
          a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
          a_imprecise = 0;
        end else if (DATA_IN_0_PRECISION_1 == 0) begin
          a_precise_2 = 0;
          a_imprecise = 0;
        end else begin
          a_precise_2[2:2-DATA_IN_0_PRECISION_1+1] = data_in_0[DATA_IN_0_PRECISION_1-1:0];
          a_precise_2[2-DATA_IN_0_PRECISION_1:0] = 0;
          a_imprecise = 0;
        end
      end
    end	
    else if (DATA_IN_0_PRECISION_INT == 4) begin  //boundary condition when integer bits are exactly 4
      a_precise_1 = data_in_0[DATA_IN_0_PRECISION_0-1:DATA_IN_0_PRECISION_1];
      if (DATA_IN_0_PRECISION_1 > 3) begin
        a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
        a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
        a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
      end else if (DATA_IN_0_PRECISION_1 == 3) begin
        a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
        a_imprecise = 0;
      end else if (DATA_IN_0_PRECISION_1 == 0) begin
        a_precise_2 = 0;
        a_imprecise = 0;
      end else begin
        a_precise_2[2:2-DATA_IN_0_PRECISION_1+1] = data_in_0[DATA_IN_0_PRECISION_1-1:0];
        a_precise_2[2-DATA_IN_0_PRECISION_1:0] = 0;
        a_imprecise = 0;
      end
    end	
    else if (DATA_IN_0_PRECISION_INT == 0) begin  //boundary condition when there are no integer bits
      a_precise_1 = 0;
      if (DATA_IN_0_PRECISION_1 > 3) begin
        a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
        a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
        a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
      end else if (DATA_IN_0_PRECISION_1 == 3) begin
        a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
        a_imprecise = 0;
      end else if (DATA_IN_0_PRECISION_1 == 0) begin
        a_precise_2 = 0;
        a_imprecise = 0;
      end else begin
        a_precise_2[2:2-DATA_IN_0_PRECISION_1+1] = data_in_0[DATA_IN_0_PRECISION_1-1:0];
        a_precise_2[2-DATA_IN_0_PRECISION_1:0] = 0;
        a_imprecise = 0;
      end
    end else begin  //condition when integer bits are less than 4
      a_precise_1[DATA_IN_0_PRECISION_INT-1:0] = data_in_0[DATA_IN_0_PRECISION_0-1:DATA_IN_0_PRECISION_1];
      a_precise_1[3:DATA_IN_0_PRECISION_INT] = 0;
      if (DATA_IN_0_PRECISION_1 > 3) begin
        a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
        a_imprecise[DATA_IN_0_PRECISION_1-3-1:0] = data_in_0[DATA_IN_0_PRECISION_1-3-1:0];
        a_imprecise[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3] = 0;
      end else if (DATA_IN_0_PRECISION_1 == 3) begin
        a_precise_2 = data_in_0[DATA_IN_0_PRECISION_1-1:DATA_IN_0_PRECISION_1-3];
        a_imprecise = 0;
      end else if (DATA_IN_0_PRECISION_1 == 0) begin
        a_precise_2 = 0;
        a_imprecise = 0;
      end else begin
        a_precise_2[2:2-DATA_IN_0_PRECISION_1+1] = data_in_0[DATA_IN_0_PRECISION_1-1:0];
        a_precise_2[2-DATA_IN_0_PRECISION_1:0] = 0;
        a_imprecise = 0;
      end
    end
  end

  //fetching exponential value of a_precise_1 from LUT  
  integer_lut_16 integer_lut_inst (
      .address (a_precise_1),
      .data_out(exp_precise_1)
  );

  //fetching exponential value of a_precise_2 from LUT
  fractional_lut_16 fractional_lut_inst (
      .address (a_precise_2),
      .data_out(exp_precise_2)
  );

  //calculating exponential value of a_imprecise using series approximator
  fixed_series_approx #(
      .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_1),
      .DATA_OUT_0_PRECISION_0(IMPRECISE_PRECISION_0 - 1)
  ) series_approx_inst (
      .data_in_0 (a_imprecise),
      .data_out_0(exp_imprecise[IMPRECISE_PRECISION_0-2 : 0])
  );

  assign exp_imprecise[IMPRECISE_PRECISION_0-1] = 0;
  assign product = exp_precise_1 * exp_precise_2 * exp_imprecise;  //final multiplication of parts

  //rounding of output result
  fixed_round #(
      .IN_WIDTH(PDT_WIDTH),
      .IN_FRAC_WIDTH(PDT_WIDTH - 3),
      .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  ) fixed_round_inst1 (
      .data_in (product),
      .data_out(data_out_0)
  );

endmodule
