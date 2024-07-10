`timescale 1ns / 1ps

module fixed_softplus #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0       = 16,  //PRECISION BASED ON THIS
    parameter DATA_IN_0_PRECISION_1       = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,   //PARALLELISM BASED ON THIS
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 * 2,
    parameter DATA_OUT_0_PRECISION_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter INPLACE = 0
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

  localparam WL = DATA_IN_0_PRECISION_0;

  //polynomial expansion coefficients

  //boundary -4<=input<-2
  const logic signed [WL-1:0] a2_1 = 16'h030b;
  ;  //in floating_point = 0.0238 <<15 => =16'd779
  const logic signed [WL-1:0] a1_1 = 16'h18ef;  //in floating_point = 0.1948 => =16'd6383
  const logic signed [WL-1:0] a0_1 = 16'h358e;  //in floating_point = 0.4184 => =16'd13710

  //boundary -2<=input<0                 
  const logic signed [WL-1:0] a2_2 = 16'h0c67;  //in floating_point = 0.0969
  const logic signed [WL-1:0] a1_2 = 16'h3c68;  //in floating_point = 0.472
  const logic signed [WL-1:0] a0_2 = 16'h581e;  //in floating_point = 0.68844

  //boundary 0<=input<2                          
  const logic signed [WL-1:0] a2_3 = 16'h0c67;  //in floating_point = 0.0969
  const logic signed [WL-1:0] a1_3 = 16'h4397;  //in floating_point = 0.528
  const logic signed [WL-1:0] a0_3 = 16'h581e;  //in floating_point = 0.68844

  //boundary 2<=input<=4                       
  const logic signed [WL-1:0] a2_4 = 16'h030b;  //in floating_point = 0.0238
  const logic signed [WL-1:0] a1_4 = 16'h6710;  //in floating_point = 0.8052
  const logic signed [WL-1:0] a0_4 = 16'h358e;  //in floating_point = 0.4184

  logic signed [WL-1:0] x_fxp;
  logic signed [(WL*2)-1:0] z;
  logic data_out_valid1, data_out_valid2;


  generate
    for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0; i = i + 1) begin : fixed_softplus
      always_ff @(posedge clk) begin
        if (rst) begin
          z <= 0;
          data_out_valid1 <= 0;
        end else if (data_out_0_ready && !data_in_0_valid) begin
          data_out_valid1 <= 0;
          z <= 0;
        end else if (data_out_0_ready && data_in_0_valid) begin
          data_out_valid1 <= 1;
          x_fxp <= $signed(data_in_0[i]);

          if (x_fxp < -4) begin  //x<-4
            z <= 0;
          end else if ((x_fxp >= -4) && (x_fxp < -2)) begin  //1st segment-4<=x<-2
            z <= (((a2_1 * x_fxp) + a1_1) * x_fxp) + a0_1;
          end else if ((x_fxp >= -2) && (x_fxp < 0)) begin  //2nd segment:-2<=x<0
            z <= (((a2_2 * x_fxp) + a1_2) * x_fxp) + a0_2;
          end else if (x_fxp >= 0 && x_fxp < 2) begin  //3rd segment: 0<=x<2 
            z <= (((a2_3 * x_fxp) + a1_3) * x_fxp) + a0_3;
          end else if (x_fxp >= 2 && x_fxp <= 4) begin  //4th segment:2<=x<=4 
            z <= (((a2_4 * x_fxp) + a1_4) * x_fxp) + a0_4;
          end else  //if (x > 4)         
            z <= x_fxp << 15;
        end else data_out_valid1 <= 0;
      end

      assign data_out_0[i] = z;
      assign data_out_0_valid = data_out_valid1;
      assign data_in_0_ready = data_out_0_ready;

    end
  endgenerate

endmodule
