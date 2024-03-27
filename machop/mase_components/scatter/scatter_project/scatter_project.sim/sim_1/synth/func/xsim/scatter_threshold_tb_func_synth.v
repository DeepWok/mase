// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2019.2 (lin64) Build 2708876 Wed Nov  6 21:39:14 MST 2019
// Date        : Wed Mar 27 17:29:30 2024
// Host        : ee-beholder0.ee.ic.ac.uk running 64-bit CentOS Linux release 7.9.2009 (Core)
// Command     : write_verilog -mode funcsim -nolib -force -file
//               /home/aw1223/new/mase/machop/mase_components/scatter/scatter_project/scatter_project.sim/sim_1/synth/func/xsim/scatter_threshold_tb_func_synth.v
// Design      : scatter_threshold
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xcu280-fsvh2892-2L-e
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* DESIGN = "1" *) (* HIGH_SLOTS = "2" *) (* PRECISION = "8" *) 
(* TENSOR_SIZE_DIM = "8" *) (* THRESHOLD = "6" *) 
(* NotValidForBitStream *)
module scatter_threshold
   (clk,
    rst,
    \data_in[7] ,
    \data_in[6] ,
    \data_in[5] ,
    \data_in[4] ,
    \data_in[3] ,
    \data_in[2] ,
    \data_in[1] ,
    \data_in[0] ,
    \o_high_precision[7] ,
    \o_high_precision[6] ,
    \o_high_precision[5] ,
    \o_high_precision[4] ,
    \o_high_precision[3] ,
    \o_high_precision[2] ,
    \o_high_precision[1] ,
    \o_high_precision[0] ,
    \o_low_precision[7] ,
    \o_low_precision[6] ,
    \o_low_precision[5] ,
    \o_low_precision[4] ,
    \o_low_precision[3] ,
    \o_low_precision[2] ,
    \o_low_precision[1] ,
    \o_low_precision[0] );
  input clk;
  input rst;
  input [7:0]\data_in[7] ;
  input [7:0]\data_in[6] ;
  input [7:0]\data_in[5] ;
  input [7:0]\data_in[4] ;
  input [7:0]\data_in[3] ;
  input [7:0]\data_in[2] ;
  input [7:0]\data_in[1] ;
  input [7:0]\data_in[0] ;
  output [7:0]\o_high_precision[7] ;
  output [7:0]\o_high_precision[6] ;
  output [7:0]\o_high_precision[5] ;
  output [7:0]\o_high_precision[4] ;
  output [7:0]\o_high_precision[3] ;
  output [7:0]\o_high_precision[2] ;
  output [7:0]\o_high_precision[1] ;
  output [7:0]\o_high_precision[0] ;
  output [7:0]\o_low_precision[7] ;
  output [7:0]\o_low_precision[6] ;
  output [7:0]\o_low_precision[5] ;
  output [7:0]\o_low_precision[4] ;
  output [7:0]\o_low_precision[3] ;
  output [7:0]\o_low_precision[2] ;
  output [7:0]\o_low_precision[1] ;
  output [7:0]\o_low_precision[0] ;

  wire [7:0]\data_in[0] ;
  wire [7:0]\data_in[1] ;
  wire [7:0]\data_in[2] ;
  wire [7:0]\data_in[3] ;
  wire [7:0]\data_in[4] ;
  wire [7:0]\data_in[5] ;
  wire [7:0]\data_in[6] ;
  wire [7:0]\data_in[7] ;
  wire [7:0]high_precision_req_vec;
  wire [7:0]\o_high_precision[0] ;
  wire \o_high_precision[0][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[0][7]_INST_0_i_2_n_0 ;
  wire [7:0]\o_high_precision[0]_OBUF ;
  wire [7:0]\o_high_precision[1] ;
  wire \o_high_precision[1][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[1][7]_INST_0_i_2_n_0 ;
  wire [7:0]\o_high_precision[1]_OBUF ;
  wire [7:0]\o_high_precision[2] ;
  wire \o_high_precision[2][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[2][7]_INST_0_i_2_n_0 ;
  wire [7:0]\o_high_precision[2]_OBUF ;
  wire [7:0]\o_high_precision[3] ;
  wire \o_high_precision[3][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[3][7]_INST_0_i_2_n_0 ;
  wire [7:0]\o_high_precision[3]_OBUF ;
  wire [7:0]\o_high_precision[4] ;
  wire \o_high_precision[4][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[4][7]_INST_0_i_10_n_0 ;
  wire \o_high_precision[4][7]_INST_0_i_11_n_0 ;
  wire \o_high_precision[4][7]_INST_0_i_12_n_0 ;
  wire \o_high_precision[4][7]_INST_0_i_7_n_0 ;
  wire \o_high_precision[4][7]_INST_0_i_8_n_0 ;
  wire \o_high_precision[4][7]_INST_0_i_9_n_0 ;
  wire [7:0]\o_high_precision[4]_OBUF ;
  wire [7:0]\o_high_precision[5] ;
  wire \o_high_precision[5][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[5][7]_INST_0_i_3_n_0 ;
  wire [7:0]\o_high_precision[5]_OBUF ;
  wire [7:0]\o_high_precision[6] ;
  wire \o_high_precision[6][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[6][7]_INST_0_i_3_n_0 ;
  wire \o_high_precision[6][7]_INST_0_i_4_n_0 ;
  wire \o_high_precision[6][7]_INST_0_i_5_n_0 ;
  wire \o_high_precision[6][7]_INST_0_i_6_n_0 ;
  wire \o_high_precision[6][7]_INST_0_i_7_n_0 ;
  wire \o_high_precision[6][7]_INST_0_i_8_n_0 ;
  wire [7:0]\o_high_precision[6]_OBUF ;
  wire [7:0]\o_high_precision[7] ;
  wire \o_high_precision[7][0]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][1]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][2]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][3]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][4]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][5]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][6]_INST_0_i_2_n_0 ;
  wire \o_high_precision[7][7]_INST_0_i_10_n_0 ;
  wire \o_high_precision[7][7]_INST_0_i_5_n_0 ;
  wire \o_high_precision[7][7]_INST_0_i_6_n_0 ;
  wire \o_high_precision[7][7]_INST_0_i_7_n_0 ;
  wire \o_high_precision[7][7]_INST_0_i_8_n_0 ;
  wire \o_high_precision[7][7]_INST_0_i_9_n_0 ;
  wire [7:0]\o_high_precision[7]_OBUF ;
  wire [7:0]\o_low_precision[0] ;
  wire [7:0]\o_low_precision[0]_OBUF ;
  wire [7:0]\o_low_precision[1] ;
  wire [7:0]\o_low_precision[1]_OBUF ;
  wire [7:0]\o_low_precision[2] ;
  wire [7:0]\o_low_precision[2]_OBUF ;
  wire [7:0]\o_low_precision[3] ;
  wire [7:0]\o_low_precision[3]_OBUF ;
  wire [7:0]\o_low_precision[4] ;
  wire [7:0]\o_low_precision[4]_OBUF ;
  wire [7:0]\o_low_precision[5] ;
  wire [7:0]\o_low_precision[5]_OBUF ;
  wire [7:0]\o_low_precision[6] ;
  wire [7:0]\o_low_precision[6]_OBUF ;
  wire [7:0]\o_low_precision[7] ;
  wire [7:0]\o_low_precision[7]_OBUF ;
  wire [6:5]output_mask;

  OBUF \o_high_precision[0][0]_INST_0 
       (.I(\o_high_precision[0]_OBUF [0]),
        .O(\o_high_precision[0] [0]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [0]));
  IBUF \o_high_precision[0][0]_INST_0_i_2 
       (.I(\data_in[0] [0]),
        .O(\o_high_precision[0][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][1]_INST_0 
       (.I(\o_high_precision[0]_OBUF [1]),
        .O(\o_high_precision[0] [1]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [1]));
  IBUF \o_high_precision[0][1]_INST_0_i_2 
       (.I(\data_in[0] [1]),
        .O(\o_high_precision[0][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][2]_INST_0 
       (.I(\o_high_precision[0]_OBUF [2]),
        .O(\o_high_precision[0] [2]));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [2]));
  IBUF \o_high_precision[0][2]_INST_0_i_2 
       (.I(\data_in[0] [2]),
        .O(\o_high_precision[0][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][3]_INST_0 
       (.I(\o_high_precision[0]_OBUF [3]),
        .O(\o_high_precision[0] [3]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [3]));
  IBUF \o_high_precision[0][3]_INST_0_i_2 
       (.I(\data_in[0] [3]),
        .O(\o_high_precision[0][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][4]_INST_0 
       (.I(\o_high_precision[0]_OBUF [4]),
        .O(\o_high_precision[0] [4]));
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [4]));
  IBUF \o_high_precision[0][4]_INST_0_i_2 
       (.I(\data_in[0] [4]),
        .O(\o_high_precision[0][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][5]_INST_0 
       (.I(\o_high_precision[0]_OBUF [5]),
        .O(\o_high_precision[0] [5]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [5]));
  IBUF \o_high_precision[0][5]_INST_0_i_2 
       (.I(\data_in[0] [5]),
        .O(\o_high_precision[0][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][6]_INST_0 
       (.I(\o_high_precision[0]_OBUF [6]),
        .O(\o_high_precision[0] [6]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [6]));
  IBUF \o_high_precision[0][6]_INST_0_i_2 
       (.I(\data_in[0] [6]),
        .O(\o_high_precision[0][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[0][7]_INST_0 
       (.I(\o_high_precision[0]_OBUF [7]),
        .O(\o_high_precision[0] [7]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[0][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[0]),
        .I1(\o_high_precision[0][7]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[0]_OBUF [7]));
  IBUF \o_high_precision[0][7]_INST_0_i_2 
       (.I(\data_in[0] [7]),
        .O(\o_high_precision[0][7]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][0]_INST_0 
       (.I(\o_high_precision[1]_OBUF [0]),
        .O(\o_high_precision[1] [0]));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][0]_INST_0_i_1 
       (.I0(\o_high_precision[1][0]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [0]));
  IBUF \o_high_precision[1][0]_INST_0_i_2 
       (.I(\data_in[1] [0]),
        .O(\o_high_precision[1][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][1]_INST_0 
       (.I(\o_high_precision[1]_OBUF [1]),
        .O(\o_high_precision[1] [1]));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][1]_INST_0_i_1 
       (.I0(\o_high_precision[1][1]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [1]));
  IBUF \o_high_precision[1][1]_INST_0_i_2 
       (.I(\data_in[1] [1]),
        .O(\o_high_precision[1][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][2]_INST_0 
       (.I(\o_high_precision[1]_OBUF [2]),
        .O(\o_high_precision[1] [2]));
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][2]_INST_0_i_1 
       (.I0(\o_high_precision[1][2]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [2]));
  IBUF \o_high_precision[1][2]_INST_0_i_2 
       (.I(\data_in[1] [2]),
        .O(\o_high_precision[1][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][3]_INST_0 
       (.I(\o_high_precision[1]_OBUF [3]),
        .O(\o_high_precision[1] [3]));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][3]_INST_0_i_1 
       (.I0(\o_high_precision[1][3]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [3]));
  IBUF \o_high_precision[1][3]_INST_0_i_2 
       (.I(\data_in[1] [3]),
        .O(\o_high_precision[1][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][4]_INST_0 
       (.I(\o_high_precision[1]_OBUF [4]),
        .O(\o_high_precision[1] [4]));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][4]_INST_0_i_1 
       (.I0(\o_high_precision[1][4]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [4]));
  IBUF \o_high_precision[1][4]_INST_0_i_2 
       (.I(\data_in[1] [4]),
        .O(\o_high_precision[1][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][5]_INST_0 
       (.I(\o_high_precision[1]_OBUF [5]),
        .O(\o_high_precision[1] [5]));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][5]_INST_0_i_1 
       (.I0(\o_high_precision[1][5]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [5]));
  IBUF \o_high_precision[1][5]_INST_0_i_2 
       (.I(\data_in[1] [5]),
        .O(\o_high_precision[1][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][6]_INST_0 
       (.I(\o_high_precision[1]_OBUF [6]),
        .O(\o_high_precision[1] [6]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][6]_INST_0_i_1 
       (.I0(\o_high_precision[1][6]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [6]));
  IBUF \o_high_precision[1][6]_INST_0_i_2 
       (.I(\data_in[1] [6]),
        .O(\o_high_precision[1][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[1][7]_INST_0 
       (.I(\o_high_precision[1]_OBUF [7]),
        .O(\o_high_precision[1] [7]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[1][7]_INST_0_i_1 
       (.I0(\o_high_precision[1][7]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_high_precision[1]_OBUF [7]));
  IBUF \o_high_precision[1][7]_INST_0_i_2 
       (.I(\data_in[1] [7]),
        .O(\o_high_precision[1][7]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][0]_INST_0 
       (.I(\o_high_precision[2]_OBUF [0]),
        .O(\o_high_precision[2] [0]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [0]));
  IBUF \o_high_precision[2][0]_INST_0_i_2 
       (.I(\data_in[2] [0]),
        .O(\o_high_precision[2][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][1]_INST_0 
       (.I(\o_high_precision[2]_OBUF [1]),
        .O(\o_high_precision[2] [1]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [1]));
  IBUF \o_high_precision[2][1]_INST_0_i_2 
       (.I(\data_in[2] [1]),
        .O(\o_high_precision[2][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][2]_INST_0 
       (.I(\o_high_precision[2]_OBUF [2]),
        .O(\o_high_precision[2] [2]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [2]));
  IBUF \o_high_precision[2][2]_INST_0_i_2 
       (.I(\data_in[2] [2]),
        .O(\o_high_precision[2][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][3]_INST_0 
       (.I(\o_high_precision[2]_OBUF [3]),
        .O(\o_high_precision[2] [3]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [3]));
  IBUF \o_high_precision[2][3]_INST_0_i_2 
       (.I(\data_in[2] [3]),
        .O(\o_high_precision[2][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][4]_INST_0 
       (.I(\o_high_precision[2]_OBUF [4]),
        .O(\o_high_precision[2] [4]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [4]));
  IBUF \o_high_precision[2][4]_INST_0_i_2 
       (.I(\data_in[2] [4]),
        .O(\o_high_precision[2][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][5]_INST_0 
       (.I(\o_high_precision[2]_OBUF [5]),
        .O(\o_high_precision[2] [5]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [5]));
  IBUF \o_high_precision[2][5]_INST_0_i_2 
       (.I(\data_in[2] [5]),
        .O(\o_high_precision[2][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][6]_INST_0 
       (.I(\o_high_precision[2]_OBUF [6]),
        .O(\o_high_precision[2] [6]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [6]));
  IBUF \o_high_precision[2][6]_INST_0_i_2 
       (.I(\data_in[2] [6]),
        .O(\o_high_precision[2][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[2][7]_INST_0 
       (.I(\o_high_precision[2]_OBUF [7]),
        .O(\o_high_precision[2] [7]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'h7000)) 
    \o_high_precision[2][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][7]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[2]_OBUF [7]));
  IBUF \o_high_precision[2][7]_INST_0_i_2 
       (.I(\data_in[2] [7]),
        .O(\o_high_precision[2][7]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][0]_INST_0 
       (.I(\o_high_precision[3]_OBUF [0]),
        .O(\o_high_precision[3] [0]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [0]));
  IBUF \o_high_precision[3][0]_INST_0_i_2 
       (.I(\data_in[3] [0]),
        .O(\o_high_precision[3][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][1]_INST_0 
       (.I(\o_high_precision[3]_OBUF [1]),
        .O(\o_high_precision[3] [1]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [1]));
  IBUF \o_high_precision[3][1]_INST_0_i_2 
       (.I(\data_in[3] [1]),
        .O(\o_high_precision[3][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][2]_INST_0 
       (.I(\o_high_precision[3]_OBUF [2]),
        .O(\o_high_precision[3] [2]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [2]));
  IBUF \o_high_precision[3][2]_INST_0_i_2 
       (.I(\data_in[3] [2]),
        .O(\o_high_precision[3][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][3]_INST_0 
       (.I(\o_high_precision[3]_OBUF [3]),
        .O(\o_high_precision[3] [3]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [3]));
  IBUF \o_high_precision[3][3]_INST_0_i_2 
       (.I(\data_in[3] [3]),
        .O(\o_high_precision[3][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][4]_INST_0 
       (.I(\o_high_precision[3]_OBUF [4]),
        .O(\o_high_precision[3] [4]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [4]));
  IBUF \o_high_precision[3][4]_INST_0_i_2 
       (.I(\data_in[3] [4]),
        .O(\o_high_precision[3][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][5]_INST_0 
       (.I(\o_high_precision[3]_OBUF [5]),
        .O(\o_high_precision[3] [5]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [5]));
  IBUF \o_high_precision[3][5]_INST_0_i_2 
       (.I(\data_in[3] [5]),
        .O(\o_high_precision[3][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][6]_INST_0 
       (.I(\o_high_precision[3]_OBUF [6]),
        .O(\o_high_precision[3] [6]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [6]));
  IBUF \o_high_precision[3][6]_INST_0_i_2 
       (.I(\data_in[3] [6]),
        .O(\o_high_precision[3][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[3][7]_INST_0 
       (.I(\o_high_precision[3]_OBUF [7]),
        .O(\o_high_precision[3] [7]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'h17000000)) 
    \o_high_precision[3][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][7]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[3]_OBUF [7]));
  IBUF \o_high_precision[3][7]_INST_0_i_2 
       (.I(\data_in[3] [7]),
        .O(\o_high_precision[3][7]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][0]_INST_0 
       (.I(\o_high_precision[4]_OBUF [0]),
        .O(\o_high_precision[4] [0]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [0]));
  IBUF \o_high_precision[4][0]_INST_0_i_2 
       (.I(\data_in[4] [0]),
        .O(\o_high_precision[4][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][1]_INST_0 
       (.I(\o_high_precision[4]_OBUF [1]),
        .O(\o_high_precision[4] [1]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [1]));
  IBUF \o_high_precision[4][1]_INST_0_i_2 
       (.I(\data_in[4] [1]),
        .O(\o_high_precision[4][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][2]_INST_0 
       (.I(\o_high_precision[4]_OBUF [2]),
        .O(\o_high_precision[4] [2]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [2]));
  IBUF \o_high_precision[4][2]_INST_0_i_2 
       (.I(\data_in[4] [2]),
        .O(\o_high_precision[4][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][3]_INST_0 
       (.I(\o_high_precision[4]_OBUF [3]),
        .O(\o_high_precision[4] [3]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [3]));
  IBUF \o_high_precision[4][3]_INST_0_i_2 
       (.I(\data_in[4] [3]),
        .O(\o_high_precision[4][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][4]_INST_0 
       (.I(\o_high_precision[4]_OBUF [4]),
        .O(\o_high_precision[4] [4]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [4]));
  IBUF \o_high_precision[4][4]_INST_0_i_2 
       (.I(\data_in[4] [4]),
        .O(\o_high_precision[4][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][5]_INST_0 
       (.I(\o_high_precision[4]_OBUF [5]),
        .O(\o_high_precision[4] [5]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [5]));
  IBUF \o_high_precision[4][5]_INST_0_i_2 
       (.I(\data_in[4] [5]),
        .O(\o_high_precision[4][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][6]_INST_0 
       (.I(\o_high_precision[4]_OBUF [6]),
        .O(\o_high_precision[4] [6]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4]_OBUF [6]));
  IBUF \o_high_precision[4][6]_INST_0_i_2 
       (.I(\data_in[4] [6]),
        .O(\o_high_precision[4][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[4][7]_INST_0 
       (.I(\o_high_precision[4]_OBUF [7]),
        .O(\o_high_precision[4] [7]));
  LUT6 #(
    .INIT(64'h0002022A00000000)) 
    \o_high_precision[4][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][7]_INST_0_i_7_n_0 ),
        .O(\o_high_precision[4]_OBUF [7]));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[4][7]_INST_0_i_10 
       (.I0(\o_high_precision[0][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[0][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[0][7]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[0][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[0][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4][7]_INST_0_i_10_n_0 ));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[4][7]_INST_0_i_11 
       (.I0(\o_high_precision[2][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[2][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[2][7]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[2][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[2][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4][7]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[4][7]_INST_0_i_12 
       (.I0(\o_high_precision[3][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[3][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[3][7]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[3][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[3][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4][7]_INST_0_i_12_n_0 ));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[4][7]_INST_0_i_2 
       (.I0(\o_high_precision[4][7]_INST_0_i_8_n_0 ),
        .I1(\o_high_precision[4][7]_INST_0_i_7_n_0 ),
        .I2(\o_high_precision[4][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[4][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[4][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[4]));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[4][7]_INST_0_i_3 
       (.I0(\o_high_precision[4][7]_INST_0_i_9_n_0 ),
        .I1(\o_high_precision[1][7]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[1][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[1][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[1][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[1]));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[4][7]_INST_0_i_4 
       (.I0(\o_high_precision[4][7]_INST_0_i_10_n_0 ),
        .I1(\o_high_precision[0][7]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[0][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[0][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[0][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[0]));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[4][7]_INST_0_i_5 
       (.I0(\o_high_precision[4][7]_INST_0_i_11_n_0 ),
        .I1(\o_high_precision[2][7]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[2][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[2][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[2][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[2]));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[4][7]_INST_0_i_6 
       (.I0(\o_high_precision[4][7]_INST_0_i_12_n_0 ),
        .I1(\o_high_precision[3][7]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[3][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[3][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[3][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[3]));
  IBUF \o_high_precision[4][7]_INST_0_i_7 
       (.I(\data_in[4] [7]),
        .O(\o_high_precision[4][7]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[4][7]_INST_0_i_8 
       (.I0(\o_high_precision[4][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[4][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[4][7]_INST_0_i_7_n_0 ),
        .I3(\o_high_precision[4][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[4][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4][7]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[4][7]_INST_0_i_9 
       (.I0(\o_high_precision[1][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[1][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[1][7]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[1][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[1][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[4][7]_INST_0_i_9_n_0 ));
  OBUF \o_high_precision[5][0]_INST_0 
       (.I(\o_high_precision[5]_OBUF [0]),
        .O(\o_high_precision[5] [0]));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][0]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [0]));
  IBUF \o_high_precision[5][0]_INST_0_i_2 
       (.I(\data_in[5] [0]),
        .O(\o_high_precision[5][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][1]_INST_0 
       (.I(\o_high_precision[5]_OBUF [1]),
        .O(\o_high_precision[5] [1]));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][1]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [1]));
  IBUF \o_high_precision[5][1]_INST_0_i_2 
       (.I(\data_in[5] [1]),
        .O(\o_high_precision[5][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][2]_INST_0 
       (.I(\o_high_precision[5]_OBUF [2]),
        .O(\o_high_precision[5] [2]));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][2]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [2]));
  IBUF \o_high_precision[5][2]_INST_0_i_2 
       (.I(\data_in[5] [2]),
        .O(\o_high_precision[5][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][3]_INST_0 
       (.I(\o_high_precision[5]_OBUF [3]),
        .O(\o_high_precision[5] [3]));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][3]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [3]));
  IBUF \o_high_precision[5][3]_INST_0_i_2 
       (.I(\data_in[5] [3]),
        .O(\o_high_precision[5][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][4]_INST_0 
       (.I(\o_high_precision[5]_OBUF [4]),
        .O(\o_high_precision[5] [4]));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][4]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [4]));
  IBUF \o_high_precision[5][4]_INST_0_i_2 
       (.I(\data_in[5] [4]),
        .O(\o_high_precision[5][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][5]_INST_0 
       (.I(\o_high_precision[5]_OBUF [5]),
        .O(\o_high_precision[5] [5]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][5]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [5]));
  IBUF \o_high_precision[5][5]_INST_0_i_2 
       (.I(\data_in[5] [5]),
        .O(\o_high_precision[5][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][6]_INST_0 
       (.I(\o_high_precision[5]_OBUF [6]),
        .O(\o_high_precision[5] [6]));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][6]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[5]_OBUF [6]));
  IBUF \o_high_precision[5][6]_INST_0_i_2 
       (.I(\data_in[5] [6]),
        .O(\o_high_precision[5][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[5][7]_INST_0 
       (.I(\o_high_precision[5]_OBUF [7]),
        .O(\o_high_precision[5] [7]));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[5][7]_INST_0_i_1 
       (.I0(output_mask[5]),
        .I1(\o_high_precision[5][7]_INST_0_i_3_n_0 ),
        .O(\o_high_precision[5]_OBUF [7]));
  LUT6 #(
    .INIT(64'h0001011700000000)) 
    \o_high_precision[5][7]_INST_0_i_2 
       (.I0(high_precision_req_vec[3]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[4]),
        .I5(high_precision_req_vec[5]),
        .O(output_mask[5]));
  IBUF \o_high_precision[5][7]_INST_0_i_3 
       (.I(\data_in[5] [7]),
        .O(\o_high_precision[5][7]_INST_0_i_3_n_0 ));
  OBUF \o_high_precision[6][0]_INST_0 
       (.I(\o_high_precision[6]_OBUF [0]),
        .O(\o_high_precision[6] [0]));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][0]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [0]));
  IBUF \o_high_precision[6][0]_INST_0_i_2 
       (.I(\data_in[6] [0]),
        .O(\o_high_precision[6][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][1]_INST_0 
       (.I(\o_high_precision[6]_OBUF [1]),
        .O(\o_high_precision[6] [1]));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][1]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [1]));
  IBUF \o_high_precision[6][1]_INST_0_i_2 
       (.I(\data_in[6] [1]),
        .O(\o_high_precision[6][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][2]_INST_0 
       (.I(\o_high_precision[6]_OBUF [2]),
        .O(\o_high_precision[6] [2]));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][2]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [2]));
  IBUF \o_high_precision[6][2]_INST_0_i_2 
       (.I(\data_in[6] [2]),
        .O(\o_high_precision[6][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][3]_INST_0 
       (.I(\o_high_precision[6]_OBUF [3]),
        .O(\o_high_precision[6] [3]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][3]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [3]));
  IBUF \o_high_precision[6][3]_INST_0_i_2 
       (.I(\data_in[6] [3]),
        .O(\o_high_precision[6][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][4]_INST_0 
       (.I(\o_high_precision[6]_OBUF [4]),
        .O(\o_high_precision[6] [4]));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][4]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [4]));
  IBUF \o_high_precision[6][4]_INST_0_i_2 
       (.I(\data_in[6] [4]),
        .O(\o_high_precision[6][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][5]_INST_0 
       (.I(\o_high_precision[6]_OBUF [5]),
        .O(\o_high_precision[6] [5]));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][5]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [5]));
  IBUF \o_high_precision[6][5]_INST_0_i_2 
       (.I(\data_in[6] [5]),
        .O(\o_high_precision[6][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][6]_INST_0 
       (.I(\o_high_precision[6]_OBUF [6]),
        .O(\o_high_precision[6] [6]));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][6]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6]_OBUF [6]));
  IBUF \o_high_precision[6][6]_INST_0_i_2 
       (.I(\data_in[6] [6]),
        .O(\o_high_precision[6][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[6][7]_INST_0 
       (.I(\o_high_precision[6]_OBUF [7]),
        .O(\o_high_precision[6] [7]));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \o_high_precision[6][7]_INST_0_i_1 
       (.I0(output_mask[6]),
        .I1(\o_high_precision[6][7]_INST_0_i_3_n_0 ),
        .O(\o_high_precision[6]_OBUF [7]));
  LUT6 #(
    .INIT(64'h0001011700000000)) 
    \o_high_precision[6][7]_INST_0_i_2 
       (.I0(high_precision_req_vec[5]),
        .I1(high_precision_req_vec[3]),
        .I2(\o_high_precision[6][7]_INST_0_i_4_n_0 ),
        .I3(high_precision_req_vec[4]),
        .I4(\o_high_precision[6][7]_INST_0_i_5_n_0 ),
        .I5(high_precision_req_vec[6]),
        .O(output_mask[6]));
  IBUF \o_high_precision[6][7]_INST_0_i_3 
       (.I(\data_in[6] [7]),
        .O(\o_high_precision[6][7]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \o_high_precision[6][7]_INST_0_i_4 
       (.I0(\o_high_precision[4][7]_INST_0_i_9_n_0 ),
        .I1(\o_high_precision[6][7]_INST_0_i_6_n_0 ),
        .I2(\o_high_precision[4][7]_INST_0_i_10_n_0 ),
        .I3(\o_high_precision[6][7]_INST_0_i_7_n_0 ),
        .I4(\o_high_precision[6][7]_INST_0_i_8_n_0 ),
        .I5(\o_high_precision[4][7]_INST_0_i_11_n_0 ),
        .O(\o_high_precision[6][7]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFEFFFEFFFEEEE0)) 
    \o_high_precision[6][7]_INST_0_i_5 
       (.I0(\o_high_precision[4][7]_INST_0_i_11_n_0 ),
        .I1(\o_high_precision[6][7]_INST_0_i_8_n_0 ),
        .I2(\o_high_precision[4][7]_INST_0_i_9_n_0 ),
        .I3(\o_high_precision[6][7]_INST_0_i_6_n_0 ),
        .I4(\o_high_precision[4][7]_INST_0_i_10_n_0 ),
        .I5(\o_high_precision[6][7]_INST_0_i_7_n_0 ),
        .O(\o_high_precision[6][7]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT4 #(
    .INIT(16'h0580)) 
    \o_high_precision[6][7]_INST_0_i_6 
       (.I0(\o_high_precision[1][2]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[1][0]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[1][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[1][7]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6][7]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT4 #(
    .INIT(16'h0580)) 
    \o_high_precision[6][7]_INST_0_i_7 
       (.I0(\o_high_precision[0][2]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[0][0]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[0][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[0][7]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6][7]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'h0580)) 
    \o_high_precision[6][7]_INST_0_i_8 
       (.I0(\o_high_precision[2][2]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[2][0]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[2][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[2][7]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[6][7]_INST_0_i_8_n_0 ));
  OBUF \o_high_precision[7][0]_INST_0 
       (.I(\o_high_precision[7]_OBUF [0]),
        .O(\o_high_precision[7] [0]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][0]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [0]));
  IBUF \o_high_precision[7][0]_INST_0_i_2 
       (.I(\data_in[7] [0]),
        .O(\o_high_precision[7][0]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][1]_INST_0 
       (.I(\o_high_precision[7]_OBUF [1]),
        .O(\o_high_precision[7] [1]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][1]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [1]));
  IBUF \o_high_precision[7][1]_INST_0_i_2 
       (.I(\data_in[7] [1]),
        .O(\o_high_precision[7][1]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][2]_INST_0 
       (.I(\o_high_precision[7]_OBUF [2]),
        .O(\o_high_precision[7] [2]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][2]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [2]));
  IBUF \o_high_precision[7][2]_INST_0_i_2 
       (.I(\data_in[7] [2]),
        .O(\o_high_precision[7][2]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][3]_INST_0 
       (.I(\o_high_precision[7]_OBUF [3]),
        .O(\o_high_precision[7] [3]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][3]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [3]));
  IBUF \o_high_precision[7][3]_INST_0_i_2 
       (.I(\data_in[7] [3]),
        .O(\o_high_precision[7][3]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][4]_INST_0 
       (.I(\o_high_precision[7]_OBUF [4]),
        .O(\o_high_precision[7] [4]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [4]));
  IBUF \o_high_precision[7][4]_INST_0_i_2 
       (.I(\data_in[7] [4]),
        .O(\o_high_precision[7][4]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][5]_INST_0 
       (.I(\o_high_precision[7]_OBUF [5]),
        .O(\o_high_precision[7] [5]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][5]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [5]));
  IBUF \o_high_precision[7][5]_INST_0_i_2 
       (.I(\data_in[7] [5]),
        .O(\o_high_precision[7][5]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][6]_INST_0 
       (.I(\o_high_precision[7]_OBUF [6]),
        .O(\o_high_precision[7] [6]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][6]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7]_OBUF [6]));
  IBUF \o_high_precision[7][6]_INST_0_i_2 
       (.I(\data_in[7] [6]),
        .O(\o_high_precision[7][6]_INST_0_i_2_n_0 ));
  OBUF \o_high_precision[7][7]_INST_0 
       (.I(\o_high_precision[7]_OBUF [7]),
        .O(\o_high_precision[7] [7]));
  LUT6 #(
    .INIT(64'h0004044C00000000)) 
    \o_high_precision[7][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[6]),
        .I1(high_precision_req_vec[7]),
        .I2(high_precision_req_vec[5]),
        .I3(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I4(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_7_n_0 ),
        .O(\o_high_precision[7]_OBUF [7]));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[7][7]_INST_0_i_10 
       (.I0(\o_high_precision[5][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[5][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[5][7]_INST_0_i_3_n_0 ),
        .I3(\o_high_precision[5][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[5][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7][7]_INST_0_i_10_n_0 ));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[7][7]_INST_0_i_2 
       (.I0(\o_high_precision[7][7]_INST_0_i_8_n_0 ),
        .I1(\o_high_precision[6][7]_INST_0_i_3_n_0 ),
        .I2(\o_high_precision[6][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[6][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[6][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[6]));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[7][7]_INST_0_i_3 
       (.I0(\o_high_precision[7][7]_INST_0_i_9_n_0 ),
        .I1(\o_high_precision[7][7]_INST_0_i_7_n_0 ),
        .I2(\o_high_precision[7][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[7][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[7][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[7]));
  LUT5 #(
    .INIT(32'hBAAAAEAE)) 
    \o_high_precision[7][7]_INST_0_i_4 
       (.I0(\o_high_precision[7][7]_INST_0_i_10_n_0 ),
        .I1(\o_high_precision[5][7]_INST_0_i_3_n_0 ),
        .I2(\o_high_precision[5][1]_INST_0_i_2_n_0 ),
        .I3(\o_high_precision[5][0]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[5][2]_INST_0_i_2_n_0 ),
        .O(high_precision_req_vec[5]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \o_high_precision[7][7]_INST_0_i_5 
       (.I0(high_precision_req_vec[3]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[4]),
        .O(\o_high_precision[7][7]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT5 #(
    .INIT(32'hFFFEFEE8)) 
    \o_high_precision[7][7]_INST_0_i_6 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[2]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[1]),
        .I4(high_precision_req_vec[3]),
        .O(\o_high_precision[7][7]_INST_0_i_6_n_0 ));
  IBUF \o_high_precision[7][7]_INST_0_i_7 
       (.I(\data_in[7] [7]),
        .O(\o_high_precision[7][7]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[7][7]_INST_0_i_8 
       (.I0(\o_high_precision[6][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[6][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[6][7]_INST_0_i_3_n_0 ),
        .I3(\o_high_precision[6][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[6][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7][7]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'h7FFFFFFE)) 
    \o_high_precision[7][7]_INST_0_i_9 
       (.I0(\o_high_precision[7][5]_INST_0_i_2_n_0 ),
        .I1(\o_high_precision[7][6]_INST_0_i_2_n_0 ),
        .I2(\o_high_precision[7][7]_INST_0_i_7_n_0 ),
        .I3(\o_high_precision[7][3]_INST_0_i_2_n_0 ),
        .I4(\o_high_precision[7][4]_INST_0_i_2_n_0 ),
        .O(\o_high_precision[7][7]_INST_0_i_9_n_0 ));
  OBUF \o_low_precision[0][0]_INST_0 
       (.I(\o_low_precision[0]_OBUF [0]),
        .O(\o_low_precision[0] [0]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][0]_INST_0_i_1 
       (.I0(\o_high_precision[0][0]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [0]));
  OBUF \o_low_precision[0][1]_INST_0 
       (.I(\o_low_precision[0]_OBUF [1]),
        .O(\o_low_precision[0] [1]));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][1]_INST_0_i_1 
       (.I0(\o_high_precision[0][1]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [1]));
  OBUF \o_low_precision[0][2]_INST_0 
       (.I(\o_low_precision[0]_OBUF [2]),
        .O(\o_low_precision[0] [2]));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][2]_INST_0_i_1 
       (.I0(\o_high_precision[0][2]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [2]));
  OBUF \o_low_precision[0][3]_INST_0 
       (.I(\o_low_precision[0]_OBUF [3]),
        .O(\o_low_precision[0] [3]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][3]_INST_0_i_1 
       (.I0(\o_high_precision[0][3]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [3]));
  OBUF \o_low_precision[0][4]_INST_0 
       (.I(\o_low_precision[0]_OBUF [4]),
        .O(\o_low_precision[0] [4]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][4]_INST_0_i_1 
       (.I0(\o_high_precision[0][4]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [4]));
  OBUF \o_low_precision[0][5]_INST_0 
       (.I(\o_low_precision[0]_OBUF [5]),
        .O(\o_low_precision[0] [5]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][5]_INST_0_i_1 
       (.I0(\o_high_precision[0][5]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [5]));
  OBUF \o_low_precision[0][6]_INST_0 
       (.I(\o_low_precision[0]_OBUF [6]),
        .O(\o_low_precision[0] [6]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][6]_INST_0_i_1 
       (.I0(\o_high_precision[0][6]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [6]));
  OBUF \o_low_precision[0][7]_INST_0 
       (.I(\o_low_precision[0]_OBUF [7]),
        .O(\o_low_precision[0] [7]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[0][7]_INST_0_i_1 
       (.I0(\o_high_precision[0][7]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[0]),
        .O(\o_low_precision[0]_OBUF [7]));
  OBUF \o_low_precision[1][0]_INST_0 
       (.I(\o_low_precision[1]_OBUF [0]),
        .O(\o_low_precision[1] [0]));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][0]_INST_0_i_1 
       (.I0(\o_high_precision[1][0]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [0]));
  OBUF \o_low_precision[1][1]_INST_0 
       (.I(\o_low_precision[1]_OBUF [1]),
        .O(\o_low_precision[1] [1]));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][1]_INST_0_i_1 
       (.I0(\o_high_precision[1][1]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [1]));
  OBUF \o_low_precision[1][2]_INST_0 
       (.I(\o_low_precision[1]_OBUF [2]),
        .O(\o_low_precision[1] [2]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][2]_INST_0_i_1 
       (.I0(\o_high_precision[1][2]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [2]));
  OBUF \o_low_precision[1][3]_INST_0 
       (.I(\o_low_precision[1]_OBUF [3]),
        .O(\o_low_precision[1] [3]));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][3]_INST_0_i_1 
       (.I0(\o_high_precision[1][3]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [3]));
  OBUF \o_low_precision[1][4]_INST_0 
       (.I(\o_low_precision[1]_OBUF [4]),
        .O(\o_low_precision[1] [4]));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][4]_INST_0_i_1 
       (.I0(\o_high_precision[1][4]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [4]));
  OBUF \o_low_precision[1][5]_INST_0 
       (.I(\o_low_precision[1]_OBUF [5]),
        .O(\o_low_precision[1] [5]));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][5]_INST_0_i_1 
       (.I0(\o_high_precision[1][5]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [5]));
  OBUF \o_low_precision[1][6]_INST_0 
       (.I(\o_low_precision[1]_OBUF [6]),
        .O(\o_low_precision[1] [6]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][6]_INST_0_i_1 
       (.I0(\o_high_precision[1][6]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [6]));
  OBUF \o_low_precision[1][7]_INST_0 
       (.I(\o_low_precision[1]_OBUF [7]),
        .O(\o_low_precision[1] [7]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[1][7]_INST_0_i_1 
       (.I0(\o_high_precision[1][7]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[1]),
        .O(\o_low_precision[1]_OBUF [7]));
  OBUF \o_low_precision[2][0]_INST_0 
       (.I(\o_low_precision[2]_OBUF [0]),
        .O(\o_low_precision[2] [0]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][0]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [0]));
  OBUF \o_low_precision[2][1]_INST_0 
       (.I(\o_low_precision[2]_OBUF [1]),
        .O(\o_low_precision[2] [1]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][1]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [1]));
  OBUF \o_low_precision[2][2]_INST_0 
       (.I(\o_low_precision[2]_OBUF [2]),
        .O(\o_low_precision[2] [2]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][2]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [2]));
  OBUF \o_low_precision[2][3]_INST_0 
       (.I(\o_low_precision[2]_OBUF [3]),
        .O(\o_low_precision[2] [3]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][3]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [3]));
  OBUF \o_low_precision[2][4]_INST_0 
       (.I(\o_low_precision[2]_OBUF [4]),
        .O(\o_low_precision[2] [4]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][4]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [4]));
  OBUF \o_low_precision[2][5]_INST_0 
       (.I(\o_low_precision[2]_OBUF [5]),
        .O(\o_low_precision[2] [5]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][5]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [5]));
  OBUF \o_low_precision[2][6]_INST_0 
       (.I(\o_low_precision[2]_OBUF [6]),
        .O(\o_low_precision[2] [6]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][6]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [6]));
  OBUF \o_low_precision[2][7]_INST_0 
       (.I(\o_low_precision[2]_OBUF [7]),
        .O(\o_low_precision[2] [7]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'h8F00)) 
    \o_low_precision[2][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(\o_high_precision[2][7]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[2]_OBUF [7]));
  OBUF \o_low_precision[3][0]_INST_0 
       (.I(\o_low_precision[3]_OBUF [0]),
        .O(\o_low_precision[3] [0]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][0]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [0]));
  OBUF \o_low_precision[3][1]_INST_0 
       (.I(\o_low_precision[3]_OBUF [1]),
        .O(\o_low_precision[3] [1]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][1]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [1]));
  OBUF \o_low_precision[3][2]_INST_0 
       (.I(\o_low_precision[3]_OBUF [2]),
        .O(\o_low_precision[3] [2]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][2]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [2]));
  OBUF \o_low_precision[3][3]_INST_0 
       (.I(\o_low_precision[3]_OBUF [3]),
        .O(\o_low_precision[3] [3]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][3]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [3]));
  OBUF \o_low_precision[3][4]_INST_0 
       (.I(\o_low_precision[3]_OBUF [4]),
        .O(\o_low_precision[3] [4]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][4]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [4]));
  OBUF \o_low_precision[3][5]_INST_0 
       (.I(\o_low_precision[3]_OBUF [5]),
        .O(\o_low_precision[3] [5]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][5]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [5]));
  OBUF \o_low_precision[3][6]_INST_0 
       (.I(\o_low_precision[3]_OBUF [6]),
        .O(\o_low_precision[3] [6]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][6]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [6]));
  OBUF \o_low_precision[3][7]_INST_0 
       (.I(\o_low_precision[3]_OBUF [7]),
        .O(\o_low_precision[3] [7]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'hE8FF0000)) 
    \o_low_precision[3][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[1]),
        .I1(high_precision_req_vec[0]),
        .I2(high_precision_req_vec[2]),
        .I3(high_precision_req_vec[3]),
        .I4(\o_high_precision[3][7]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[3]_OBUF [7]));
  OBUF \o_low_precision[4][0]_INST_0 
       (.I(\o_low_precision[4]_OBUF [0]),
        .O(\o_low_precision[4] [0]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][0]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][0]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [0]));
  OBUF \o_low_precision[4][1]_INST_0 
       (.I(\o_low_precision[4]_OBUF [1]),
        .O(\o_low_precision[4] [1]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][1]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][1]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [1]));
  OBUF \o_low_precision[4][2]_INST_0 
       (.I(\o_low_precision[4]_OBUF [2]),
        .O(\o_low_precision[4] [2]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][2]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][2]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [2]));
  OBUF \o_low_precision[4][3]_INST_0 
       (.I(\o_low_precision[4]_OBUF [3]),
        .O(\o_low_precision[4] [3]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][3]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][3]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [3]));
  OBUF \o_low_precision[4][4]_INST_0 
       (.I(\o_low_precision[4]_OBUF [4]),
        .O(\o_low_precision[4] [4]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][4]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][4]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [4]));
  OBUF \o_low_precision[4][5]_INST_0 
       (.I(\o_low_precision[4]_OBUF [5]),
        .O(\o_low_precision[4] [5]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][5]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][5]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [5]));
  OBUF \o_low_precision[4][6]_INST_0 
       (.I(\o_low_precision[4]_OBUF [6]),
        .O(\o_low_precision[4] [6]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][6]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][6]_INST_0_i_2_n_0 ),
        .O(\o_low_precision[4]_OBUF [6]));
  OBUF \o_low_precision[4][7]_INST_0 
       (.I(\o_low_precision[4]_OBUF [7]),
        .O(\o_low_precision[4] [7]));
  LUT6 #(
    .INIT(64'hFFFDFDD500000000)) 
    \o_low_precision[4][7]_INST_0_i_1 
       (.I0(high_precision_req_vec[4]),
        .I1(high_precision_req_vec[1]),
        .I2(high_precision_req_vec[0]),
        .I3(high_precision_req_vec[2]),
        .I4(high_precision_req_vec[3]),
        .I5(\o_high_precision[4][7]_INST_0_i_7_n_0 ),
        .O(\o_low_precision[4]_OBUF [7]));
  OBUF \o_low_precision[5][0]_INST_0 
       (.I(\o_low_precision[5]_OBUF [0]),
        .O(\o_low_precision[5] [0]));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][0]_INST_0_i_1 
       (.I0(\o_high_precision[5][0]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [0]));
  OBUF \o_low_precision[5][1]_INST_0 
       (.I(\o_low_precision[5]_OBUF [1]),
        .O(\o_low_precision[5] [1]));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][1]_INST_0_i_1 
       (.I0(\o_high_precision[5][1]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [1]));
  OBUF \o_low_precision[5][2]_INST_0 
       (.I(\o_low_precision[5]_OBUF [2]),
        .O(\o_low_precision[5] [2]));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][2]_INST_0_i_1 
       (.I0(\o_high_precision[5][2]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [2]));
  OBUF \o_low_precision[5][3]_INST_0 
       (.I(\o_low_precision[5]_OBUF [3]),
        .O(\o_low_precision[5] [3]));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][3]_INST_0_i_1 
       (.I0(\o_high_precision[5][3]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [3]));
  OBUF \o_low_precision[5][4]_INST_0 
       (.I(\o_low_precision[5]_OBUF [4]),
        .O(\o_low_precision[5] [4]));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][4]_INST_0_i_1 
       (.I0(\o_high_precision[5][4]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [4]));
  OBUF \o_low_precision[5][5]_INST_0 
       (.I(\o_low_precision[5]_OBUF [5]),
        .O(\o_low_precision[5] [5]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][5]_INST_0_i_1 
       (.I0(\o_high_precision[5][5]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [5]));
  OBUF \o_low_precision[5][6]_INST_0 
       (.I(\o_low_precision[5]_OBUF [6]),
        .O(\o_low_precision[5] [6]));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][6]_INST_0_i_1 
       (.I0(\o_high_precision[5][6]_INST_0_i_2_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [6]));
  OBUF \o_low_precision[5][7]_INST_0 
       (.I(\o_low_precision[5]_OBUF [7]),
        .O(\o_low_precision[5] [7]));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[5][7]_INST_0_i_1 
       (.I0(\o_high_precision[5][7]_INST_0_i_3_n_0 ),
        .I1(output_mask[5]),
        .O(\o_low_precision[5]_OBUF [7]));
  OBUF \o_low_precision[6][0]_INST_0 
       (.I(\o_low_precision[6]_OBUF [0]),
        .O(\o_low_precision[6] [0]));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][0]_INST_0_i_1 
       (.I0(\o_high_precision[6][0]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [0]));
  OBUF \o_low_precision[6][1]_INST_0 
       (.I(\o_low_precision[6]_OBUF [1]),
        .O(\o_low_precision[6] [1]));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][1]_INST_0_i_1 
       (.I0(\o_high_precision[6][1]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [1]));
  OBUF \o_low_precision[6][2]_INST_0 
       (.I(\o_low_precision[6]_OBUF [2]),
        .O(\o_low_precision[6] [2]));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][2]_INST_0_i_1 
       (.I0(\o_high_precision[6][2]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [2]));
  OBUF \o_low_precision[6][3]_INST_0 
       (.I(\o_low_precision[6]_OBUF [3]),
        .O(\o_low_precision[6] [3]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][3]_INST_0_i_1 
       (.I0(\o_high_precision[6][3]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [3]));
  OBUF \o_low_precision[6][4]_INST_0 
       (.I(\o_low_precision[6]_OBUF [4]),
        .O(\o_low_precision[6] [4]));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][4]_INST_0_i_1 
       (.I0(\o_high_precision[6][4]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [4]));
  OBUF \o_low_precision[6][5]_INST_0 
       (.I(\o_low_precision[6]_OBUF [5]),
        .O(\o_low_precision[6] [5]));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][5]_INST_0_i_1 
       (.I0(\o_high_precision[6][5]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [5]));
  OBUF \o_low_precision[6][6]_INST_0 
       (.I(\o_low_precision[6]_OBUF [6]),
        .O(\o_low_precision[6] [6]));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][6]_INST_0_i_1 
       (.I0(\o_high_precision[6][6]_INST_0_i_2_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [6]));
  OBUF \o_low_precision[6][7]_INST_0 
       (.I(\o_low_precision[6]_OBUF [7]),
        .O(\o_low_precision[6] [7]));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \o_low_precision[6][7]_INST_0_i_1 
       (.I0(\o_high_precision[6][7]_INST_0_i_3_n_0 ),
        .I1(output_mask[6]),
        .O(\o_low_precision[6]_OBUF [7]));
  OBUF \o_low_precision[7][0]_INST_0 
       (.I(\o_low_precision[7]_OBUF [0]),
        .O(\o_low_precision[7] [0]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][0]_INST_0_i_1 
       (.I0(\o_high_precision[7][0]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [0]));
  OBUF \o_low_precision[7][1]_INST_0 
       (.I(\o_low_precision[7]_OBUF [1]),
        .O(\o_low_precision[7] [1]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][1]_INST_0_i_1 
       (.I0(\o_high_precision[7][1]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [1]));
  OBUF \o_low_precision[7][2]_INST_0 
       (.I(\o_low_precision[7]_OBUF [2]),
        .O(\o_low_precision[7] [2]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][2]_INST_0_i_1 
       (.I0(\o_high_precision[7][2]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [2]));
  OBUF \o_low_precision[7][3]_INST_0 
       (.I(\o_low_precision[7]_OBUF [3]),
        .O(\o_low_precision[7] [3]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][3]_INST_0_i_1 
       (.I0(\o_high_precision[7][3]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [3]));
  OBUF \o_low_precision[7][4]_INST_0 
       (.I(\o_low_precision[7]_OBUF [4]),
        .O(\o_low_precision[7] [4]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][4]_INST_0_i_1 
       (.I0(\o_high_precision[7][4]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [4]));
  OBUF \o_low_precision[7][5]_INST_0 
       (.I(\o_low_precision[7]_OBUF [5]),
        .O(\o_low_precision[7] [5]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][5]_INST_0_i_1 
       (.I0(\o_high_precision[7][5]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [5]));
  OBUF \o_low_precision[7][6]_INST_0 
       (.I(\o_low_precision[7]_OBUF [6]),
        .O(\o_low_precision[7] [6]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][6]_INST_0_i_1 
       (.I0(\o_high_precision[7][6]_INST_0_i_2_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [6]));
  OBUF \o_low_precision[7][7]_INST_0 
       (.I(\o_low_precision[7]_OBUF [7]),
        .O(\o_low_precision[7] [7]));
  LUT6 #(
    .INIT(64'hAAAAAA8AAA8A8A0A)) 
    \o_low_precision[7][7]_INST_0_i_1 
       (.I0(\o_high_precision[7][7]_INST_0_i_7_n_0 ),
        .I1(high_precision_req_vec[6]),
        .I2(high_precision_req_vec[7]),
        .I3(high_precision_req_vec[5]),
        .I4(\o_high_precision[7][7]_INST_0_i_5_n_0 ),
        .I5(\o_high_precision[7][7]_INST_0_i_6_n_0 ),
        .O(\o_low_precision[7]_OBUF [7]));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

endmodule
`endif
