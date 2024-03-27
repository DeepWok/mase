// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
module scatter_threshold_synthesis(clk, rst, \data_in[15] , \data_in[14] , 
  \data_in[13] , \data_in[12] , \data_in[11] , \data_in[10] , \data_in[9] , \data_in[8] , 
  \data_in[7] , \data_in[6] , \data_in[5] , \data_in[4] , \data_in[3] , \data_in[2] , 
  \data_in[1] , \data_in[0] , \o_high_precision[15] , \o_high_precision[14] , 
  \o_high_precision[13] , \o_high_precision[12] , \o_high_precision[11] , 
  \o_high_precision[10] , \o_high_precision[9] , \o_high_precision[8] , 
  \o_high_precision[7] , \o_high_precision[6] , \o_high_precision[5] , 
  \o_high_precision[4] , \o_high_precision[3] , \o_high_precision[2] , 
  \o_high_precision[1] , \o_high_precision[0] , \o_low_precision[15] , 
  \o_low_precision[14] , \o_low_precision[13] , \o_low_precision[12] , 
  \o_low_precision[11] , \o_low_precision[10] , \o_low_precision[9] , 
  \o_low_precision[8] , \o_low_precision[7] , \o_low_precision[6] , \o_low_precision[5] , 
  \o_low_precision[4] , \o_low_precision[3] , \o_low_precision[2] , \o_low_precision[1] , 
  \o_low_precision[0] );
  input clk;
  input rst;
  input [15:0]\data_in[15] ;
  input [15:0]\data_in[14] ;
  input [15:0]\data_in[13] ;
  input [15:0]\data_in[12] ;
  input [15:0]\data_in[11] ;
  input [15:0]\data_in[10] ;
  input [15:0]\data_in[9] ;
  input [15:0]\data_in[8] ;
  input [15:0]\data_in[7] ;
  input [15:0]\data_in[6] ;
  input [15:0]\data_in[5] ;
  input [15:0]\data_in[4] ;
  input [15:0]\data_in[3] ;
  input [15:0]\data_in[2] ;
  input [15:0]\data_in[1] ;
  input [15:0]\data_in[0] ;
  output [15:0]\o_high_precision[15] ;
  output [15:0]\o_high_precision[14] ;
  output [15:0]\o_high_precision[13] ;
  output [15:0]\o_high_precision[12] ;
  output [15:0]\o_high_precision[11] ;
  output [15:0]\o_high_precision[10] ;
  output [15:0]\o_high_precision[9] ;
  output [15:0]\o_high_precision[8] ;
  output [15:0]\o_high_precision[7] ;
  output [15:0]\o_high_precision[6] ;
  output [15:0]\o_high_precision[5] ;
  output [15:0]\o_high_precision[4] ;
  output [15:0]\o_high_precision[3] ;
  output [15:0]\o_high_precision[2] ;
  output [15:0]\o_high_precision[1] ;
  output [15:0]\o_high_precision[0] ;
  output [15:0]\o_low_precision[15] ;
  output [15:0]\o_low_precision[14] ;
  output [15:0]\o_low_precision[13] ;
  output [15:0]\o_low_precision[12] ;
  output [15:0]\o_low_precision[11] ;
  output [15:0]\o_low_precision[10] ;
  output [15:0]\o_low_precision[9] ;
  output [15:0]\o_low_precision[8] ;
  output [15:0]\o_low_precision[7] ;
  output [15:0]\o_low_precision[6] ;
  output [15:0]\o_low_precision[5] ;
  output [15:0]\o_low_precision[4] ;
  output [15:0]\o_low_precision[3] ;
  output [15:0]\o_low_precision[2] ;
  output [15:0]\o_low_precision[1] ;
  output [15:0]\o_low_precision[0] ;
endmodule
