// ==============================================================
// Generated by Vitis HLS v2023.1
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
(* CORE_GENERATION_INFO="div_div,hls_ip_2023_1,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xcu250-figd2104-2L-e,HLS_INPUT_CLOCK=10.000000,HLS_INPUT_ARCH=pipeline,HLS_SYN_CLOCK=2.594500,HLS_SYN_LAT=27,HLS_SYN_TPT=1,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=1487,HLS_SYN_LUT=1130,HLS_VERSION=2023_1}" *)
// NOTE!! This div is based on the int div generated based on hls, which can only handle the 16/16 division
// NOTE!! When divisor_data == 0, quotient_data will be set to 111..111 which is -1, 
//        But at this point, this div is only used for softmax, so negelect this bug.
module fixed_div #(
    parameter IN_NUM = 8,
    parameter DIVIDEND_WIDTH = 8,
    parameter DIVISOR_WIDTH = 8,
    parameter QUOTIENT_WIDTH = 8
) (
    input logic clk,
    input logic rst,
    input logic [DIVIDEND_WIDTH-1:0] dividend_data[IN_NUM - 1:0],
    input logic dividend_data_valid,
    output logic dividend_data_ready,
    input logic [DIVISOR_WIDTH-1:0] divisor_data[IN_NUM - 1:0],
    input logic divisor_data_valid,
    output logic divisor_data_ready,
    output logic [QUOTIENT_WIDTH-1:0] quotient_data[IN_NUM - 1:0],
    output logic quotient_data_valid,
    input logic quotient_data_ready
);
  initial begin
    assert (DIVIDEND_WIDTH <= 16)
    else $fatal("DIVIDEND_WIDTH Set may cause resolution loss.");
    assert (DIVISOR_WIDTH <= 16)
    else $fatal("DIVISOR_WIDTH Set may cause resolution loss.");
    assert (QUOTIENT_WIDTH <= 16)
    else $fatal("QUOTIENT_WIDTH Set may cause resolution loss.");
  end

  logic [IN_NUM - 1:0]
      dividend_data_ready_expand, divisor_data_ready_expand, quotient_data_valid_expand;
  always_comb begin
    dividend_data_ready = dividend_data_ready_expand[0];
    divisor_data_ready  = divisor_data_ready_expand[0];
    quotient_data_valid = quotient_data_valid_expand[0];
  end
  logic [15:0] rounding_dividend[IN_NUM - 1:0];
  logic [15:0] rounding_divisor [IN_NUM - 1:0];
  logic [15:0] rounding_quotient[IN_NUM - 1:0];

  fixed_rounding #(
      .IN_SIZE(IN_NUM),
      .IN_WIDTH(DIVIDEND_WIDTH),
      .IN_FRAC_WIDTH(0),
      .OUT_WIDTH(16),
      .OUT_FRAC_WIDTH(0)
  ) dividend_round_inst (
      .data_in (dividend_data),
      .data_out(rounding_dividend)
  );
  fixed_rounding #(
      .IN_SIZE(IN_NUM),
      .IN_WIDTH(DIVISOR_WIDTH),
      .IN_FRAC_WIDTH(0),
      .OUT_WIDTH(16),
      .OUT_FRAC_WIDTH(0)
  ) divisor_round_inst (
      .data_in (divisor_data),
      .data_out(rounding_divisor)
  );

  fixed_rounding #(
      .IN_SIZE(IN_NUM),
      .IN_WIDTH(16),
      .IN_FRAC_WIDTH(0),
      .OUT_WIDTH(QUOTIENT_WIDTH),
      .OUT_FRAC_WIDTH(0)
  ) quotient_round_inst (
      .data_in (rounding_quotient),
      .data_out(quotient_data)
  );
  for (genvar i = 0; i < IN_NUM; i++) begin
    div div (
        .ap_clk(clk),
        .ap_rst(rst),
        .ap_start(1'b1),
        .ap_done(),
        .ap_idle(),
        .ap_ready(),
        .data_in_0_dout(rounding_dividend[i]),
        .data_in_0_empty_n(dividend_data_valid),
        .data_in_0_read(dividend_data_ready_expand[i]),
        .data_in_1_dout(rounding_divisor[i]),
        .data_in_1_empty_n(divisor_data_valid),
        .data_in_1_read(divisor_data_ready_expand[i]),
        .data_out_0_din(rounding_quotient[i]),
        .data_out_0_write(quotient_data_valid_expand[i]),
        .data_out_0_full_n(quotient_data_ready)
    );
  end
endmodule

// ==============================================================
// Generated by Vitis HLS v2023.1
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================


module div (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        data_in_0_dout,
        data_in_0_empty_n,
        data_in_0_read,
        data_in_1_dout,
        data_in_1_empty_n,
        data_in_1_read,
        data_out_0_din,
        data_out_0_full_n,
        data_out_0_write
);

parameter    ap_ST_fsm_pp0_stage0 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [15:0] data_in_0_dout;
input   data_in_0_empty_n;
output   data_in_0_read;
input  [15:0] data_in_1_dout;
input   data_in_1_empty_n;
output   data_in_1_read;
output  [15:0] data_out_0_din;
input   data_out_0_full_n;
output   data_out_0_write;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg data_in_0_read;
reg data_in_1_read;
reg data_out_0_write;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_enable_reg_pp0_iter3;
reg    ap_enable_reg_pp0_iter4;
reg    ap_enable_reg_pp0_iter5;
reg    ap_enable_reg_pp0_iter6;
reg    ap_enable_reg_pp0_iter7;
reg    ap_enable_reg_pp0_iter8;
reg    ap_enable_reg_pp0_iter9;
reg    ap_enable_reg_pp0_iter10;
reg    ap_enable_reg_pp0_iter11;
reg    ap_enable_reg_pp0_iter12;
reg    ap_enable_reg_pp0_iter13;
reg    ap_enable_reg_pp0_iter14;
reg    ap_enable_reg_pp0_iter15;
reg    ap_enable_reg_pp0_iter16;
reg    ap_enable_reg_pp0_iter17;
reg    ap_enable_reg_pp0_iter18;
reg    ap_enable_reg_pp0_iter19;
reg    ap_idle_pp0;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
wire    ap_block_state3_pp0_stage0_iter2;
wire    ap_block_state4_pp0_stage0_iter3;
wire    ap_block_state5_pp0_stage0_iter4;
wire    ap_block_state6_pp0_stage0_iter5;
wire    ap_block_state7_pp0_stage0_iter6;
wire    ap_block_state8_pp0_stage0_iter7;
wire    ap_block_state9_pp0_stage0_iter8;
wire    ap_block_state10_pp0_stage0_iter9;
wire    ap_block_state11_pp0_stage0_iter10;
wire    ap_block_state12_pp0_stage0_iter11;
wire    ap_block_state13_pp0_stage0_iter12;
wire    ap_block_state14_pp0_stage0_iter13;
wire    ap_block_state15_pp0_stage0_iter14;
wire    ap_block_state16_pp0_stage0_iter15;
wire    ap_block_state17_pp0_stage0_iter16;
wire    ap_block_state18_pp0_stage0_iter17;
wire    ap_block_state19_pp0_stage0_iter18;
wire    ap_block_state20_pp0_stage0_iter19;
wire    ap_block_pp0_stage0_subdone;
wire    ap_block_pp0_stage0_11001;
wire   [0:0] tmp_nbreadreq_fu_32_p3;
wire   [0:0] tmp_1_nbreadreq_fu_40_p3;
reg   [0:0] tmp_1_reg_86;
reg   [0:0] tmp_1_reg_86_pp0_iter1_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter2_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter3_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter4_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter5_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter6_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter7_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter8_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter9_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter10_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter11_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter12_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter13_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter14_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter15_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter16_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter17_reg;
reg   [0:0] tmp_1_reg_86_pp0_iter18_reg;
reg   [0:0] tmp_reg_95;
reg   [0:0] tmp_reg_95_pp0_iter1_reg;
reg   [0:0] tmp_reg_95_pp0_iter2_reg;
reg   [0:0] tmp_reg_95_pp0_iter3_reg;
reg   [0:0] tmp_reg_95_pp0_iter4_reg;
reg   [0:0] tmp_reg_95_pp0_iter5_reg;
reg   [0:0] tmp_reg_95_pp0_iter6_reg;
reg   [0:0] tmp_reg_95_pp0_iter7_reg;
reg   [0:0] tmp_reg_95_pp0_iter8_reg;
reg   [0:0] tmp_reg_95_pp0_iter9_reg;
reg   [0:0] tmp_reg_95_pp0_iter10_reg;
reg   [0:0] tmp_reg_95_pp0_iter11_reg;
reg   [0:0] tmp_reg_95_pp0_iter12_reg;
reg   [0:0] tmp_reg_95_pp0_iter13_reg;
reg   [0:0] tmp_reg_95_pp0_iter14_reg;
reg   [0:0] tmp_reg_95_pp0_iter15_reg;
reg   [0:0] tmp_reg_95_pp0_iter16_reg;
reg   [0:0] tmp_reg_95_pp0_iter17_reg;
reg   [0:0] tmp_reg_95_pp0_iter18_reg;
wire   [15:0] grp_fu_75_p2;
wire    ap_block_pp0_stage0_01001;
wire    ap_block_pp0_stage0;
reg   [0:0] ap_NS_fsm;
reg    ap_idle_pp0_0to18;
reg    ap_reset_idle_pp0;
wire    ap_enable_pp0;
wire    ap_ce_reg;

// power-on initialization
initial begin
ap_CS_fsm = 1'd1;
ap_enable_reg_pp0_iter1 = 1'b0;
ap_enable_reg_pp0_iter2 = 1'b0;
ap_enable_reg_pp0_iter3 = 1'b0;
ap_enable_reg_pp0_iter4 = 1'b0;
ap_enable_reg_pp0_iter5 = 1'b0;
ap_enable_reg_pp0_iter6 = 1'b0;
ap_enable_reg_pp0_iter7 = 1'b0;
ap_enable_reg_pp0_iter8 = 1'b0;
ap_enable_reg_pp0_iter9 = 1'b0;
ap_enable_reg_pp0_iter10 = 1'b0;
ap_enable_reg_pp0_iter11 = 1'b0;
ap_enable_reg_pp0_iter12 = 1'b0;
ap_enable_reg_pp0_iter13 = 1'b0;
ap_enable_reg_pp0_iter14 = 1'b0;
ap_enable_reg_pp0_iter15 = 1'b0;
ap_enable_reg_pp0_iter16 = 1'b0;
ap_enable_reg_pp0_iter17 = 1'b0;
ap_enable_reg_pp0_iter18 = 1'b0;
ap_enable_reg_pp0_iter19 = 1'b0;
end

div_sdiv_16ns_16ns_16_20_1 #(
    .ID( 1 ),
    .NUM_STAGE( 20 ),
    .din0_WIDTH( 16 ),
    .din1_WIDTH( 16 ),
    .dout_WIDTH( 16 ))
sdiv_16ns_16ns_16_20_1_U1(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_in_0_dout),
    .din1(data_in_1_dout),
    .ce(1'b1),
    .dout(grp_fu_75_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_enable_reg_pp0_iter1 <= ap_start;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter10 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter10 <= ap_enable_reg_pp0_iter9;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter11 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter11 <= ap_enable_reg_pp0_iter10;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter12 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter12 <= ap_enable_reg_pp0_iter11;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter13 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter13 <= ap_enable_reg_pp0_iter12;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter14 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter14 <= ap_enable_reg_pp0_iter13;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter15 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter15 <= ap_enable_reg_pp0_iter14;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter16 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter16 <= ap_enable_reg_pp0_iter15;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter17 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter17 <= ap_enable_reg_pp0_iter16;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter18 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter18 <= ap_enable_reg_pp0_iter17;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter19 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter19 <= ap_enable_reg_pp0_iter18;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter4 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter4 <= ap_enable_reg_pp0_iter3;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter5 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter5 <= ap_enable_reg_pp0_iter4;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter6 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter6 <= ap_enable_reg_pp0_iter5;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter7 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter7 <= ap_enable_reg_pp0_iter6;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter8 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter8 <= ap_enable_reg_pp0_iter7;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter9 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter9 <= ap_enable_reg_pp0_iter8;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_nbreadreq_fu_32_p3 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        tmp_1_reg_86 <= tmp_1_nbreadreq_fu_40_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b0 == ap_block_pp0_stage0_11001)) begin
        tmp_1_reg_86_pp0_iter10_reg <= tmp_1_reg_86_pp0_iter9_reg;
        tmp_1_reg_86_pp0_iter11_reg <= tmp_1_reg_86_pp0_iter10_reg;
        tmp_1_reg_86_pp0_iter12_reg <= tmp_1_reg_86_pp0_iter11_reg;
        tmp_1_reg_86_pp0_iter13_reg <= tmp_1_reg_86_pp0_iter12_reg;
        tmp_1_reg_86_pp0_iter14_reg <= tmp_1_reg_86_pp0_iter13_reg;
        tmp_1_reg_86_pp0_iter15_reg <= tmp_1_reg_86_pp0_iter14_reg;
        tmp_1_reg_86_pp0_iter16_reg <= tmp_1_reg_86_pp0_iter15_reg;
        tmp_1_reg_86_pp0_iter17_reg <= tmp_1_reg_86_pp0_iter16_reg;
        tmp_1_reg_86_pp0_iter18_reg <= tmp_1_reg_86_pp0_iter17_reg;
        tmp_1_reg_86_pp0_iter2_reg <= tmp_1_reg_86_pp0_iter1_reg;
        tmp_1_reg_86_pp0_iter3_reg <= tmp_1_reg_86_pp0_iter2_reg;
        tmp_1_reg_86_pp0_iter4_reg <= tmp_1_reg_86_pp0_iter3_reg;
        tmp_1_reg_86_pp0_iter5_reg <= tmp_1_reg_86_pp0_iter4_reg;
        tmp_1_reg_86_pp0_iter6_reg <= tmp_1_reg_86_pp0_iter5_reg;
        tmp_1_reg_86_pp0_iter7_reg <= tmp_1_reg_86_pp0_iter6_reg;
        tmp_1_reg_86_pp0_iter8_reg <= tmp_1_reg_86_pp0_iter7_reg;
        tmp_1_reg_86_pp0_iter9_reg <= tmp_1_reg_86_pp0_iter8_reg;
        tmp_reg_95_pp0_iter10_reg <= tmp_reg_95_pp0_iter9_reg;
        tmp_reg_95_pp0_iter11_reg <= tmp_reg_95_pp0_iter10_reg;
        tmp_reg_95_pp0_iter12_reg <= tmp_reg_95_pp0_iter11_reg;
        tmp_reg_95_pp0_iter13_reg <= tmp_reg_95_pp0_iter12_reg;
        tmp_reg_95_pp0_iter14_reg <= tmp_reg_95_pp0_iter13_reg;
        tmp_reg_95_pp0_iter15_reg <= tmp_reg_95_pp0_iter14_reg;
        tmp_reg_95_pp0_iter16_reg <= tmp_reg_95_pp0_iter15_reg;
        tmp_reg_95_pp0_iter17_reg <= tmp_reg_95_pp0_iter16_reg;
        tmp_reg_95_pp0_iter18_reg <= tmp_reg_95_pp0_iter17_reg;
        tmp_reg_95_pp0_iter2_reg <= tmp_reg_95_pp0_iter1_reg;
        tmp_reg_95_pp0_iter3_reg <= tmp_reg_95_pp0_iter2_reg;
        tmp_reg_95_pp0_iter4_reg <= tmp_reg_95_pp0_iter3_reg;
        tmp_reg_95_pp0_iter5_reg <= tmp_reg_95_pp0_iter4_reg;
        tmp_reg_95_pp0_iter6_reg <= tmp_reg_95_pp0_iter5_reg;
        tmp_reg_95_pp0_iter7_reg <= tmp_reg_95_pp0_iter6_reg;
        tmp_reg_95_pp0_iter8_reg <= tmp_reg_95_pp0_iter7_reg;
        tmp_reg_95_pp0_iter9_reg <= tmp_reg_95_pp0_iter8_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        tmp_1_reg_86_pp0_iter1_reg <= tmp_1_reg_86;
        tmp_reg_95 <= tmp_nbreadreq_fu_32_p3;
        tmp_reg_95_pp0_iter1_reg <= tmp_reg_95;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter19 == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_idle_pp0 == 1'b1) & (ap_start == 1'b0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter19 == 1'b0) & (ap_enable_reg_pp0_iter18 == 1'b0) & (ap_enable_reg_pp0_iter17 == 1'b0) & (ap_enable_reg_pp0_iter16 == 1'b0) & (ap_enable_reg_pp0_iter15 == 1'b0) & (ap_enable_reg_pp0_iter14 == 1'b0) & (ap_enable_reg_pp0_iter13 == 1'b0) & (ap_enable_reg_pp0_iter12 == 1'b0) & (ap_enable_reg_pp0_iter11 == 1'b0) & (ap_enable_reg_pp0_iter10 == 1'b0) & (ap_enable_reg_pp0_iter9 == 1'b0) & (ap_enable_reg_pp0_iter8 == 1'b0) & (ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter18 == 1'b0) & (ap_enable_reg_pp0_iter17 == 1'b0) & (ap_enable_reg_pp0_iter16 == 1'b0) & (ap_enable_reg_pp0_iter15 == 1'b0) & (ap_enable_reg_pp0_iter14 == 1'b0) & (ap_enable_reg_pp0_iter13 == 1'b0) & (ap_enable_reg_pp0_iter12 == 1'b0) & (ap_enable_reg_pp0_iter11 == 1'b0) & (ap_enable_reg_pp0_iter10 == 1'b0) & (ap_enable_reg_pp0_iter9 == 1'b0) & (ap_enable_reg_pp0_iter8 == 1'b0) & (ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0_0to18 = 1'b1;
    end else begin
        ap_idle_pp0_0to18 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (ap_idle_pp0_0to18 == 1'b1))) begin
        ap_reset_idle_pp0 = 1'b1;
    end else begin
        ap_reset_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_1_nbreadreq_fu_40_p3 == 1'd1) & (tmp_nbreadreq_fu_32_p3 == 1'd1) & (data_in_0_empty_n == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        data_in_0_read = 1'b1;
    end else begin
        data_in_0_read = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (tmp_1_nbreadreq_fu_40_p3 == 1'd1) & (tmp_nbreadreq_fu_32_p3 == 1'd1) & (data_in_1_empty_n == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        data_in_1_read = 1'b1;
    end else begin
        data_in_1_read = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter19 == 1'b1) & (tmp_reg_95_pp0_iter18_reg == 1'd1) & (tmp_1_reg_86_pp0_iter18_reg == 1'd1) & (data_out_0_full_n == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        data_out_0_write = 1'b1;
    end else begin
        data_out_0_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_01001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_subdone = ~(1'b1 == 1'b1);

assign ap_block_state10_pp0_stage0_iter9 = ~(1'b1 == 1'b1);

assign ap_block_state11_pp0_stage0_iter10 = ~(1'b1 == 1'b1);

assign ap_block_state12_pp0_stage0_iter11 = ~(1'b1 == 1'b1);

assign ap_block_state13_pp0_stage0_iter12 = ~(1'b1 == 1'b1);

assign ap_block_state14_pp0_stage0_iter13 = ~(1'b1 == 1'b1);

assign ap_block_state15_pp0_stage0_iter14 = ~(1'b1 == 1'b1);

assign ap_block_state16_pp0_stage0_iter15 = ~(1'b1 == 1'b1);

assign ap_block_state17_pp0_stage0_iter16 = ~(1'b1 == 1'b1);

assign ap_block_state18_pp0_stage0_iter17 = ~(1'b1 == 1'b1);

assign ap_block_state19_pp0_stage0_iter18 = ~(1'b1 == 1'b1);

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state20_pp0_stage0_iter19 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state3_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state4_pp0_stage0_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage0_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage0_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage0_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage0_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage0_iter8 = ~(1'b1 == 1'b1);

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start;

assign data_out_0_din = grp_fu_75_p2;

assign tmp_1_nbreadreq_fu_40_p3 = data_in_1_empty_n;

assign tmp_nbreadreq_fu_32_p3 = data_in_0_empty_n;

endmodule //div

module div_sdiv_16ns_16ns_16_20_1_divider #(
    parameter in0_WIDTH = 32,
    in1_WIDTH = 32,
    out_WIDTH = 32
) (
    input                       clk,
    input                       reset,
    input                       ce,
    input       [in0_WIDTH-1:0] dividend,
    input       [in1_WIDTH-1:0] divisor,
    input       [          1:0] sign_i,
    output wire [          1:0] sign_o,
    output wire [out_WIDTH-1:0] quot,
    output wire [out_WIDTH-1:0] remd
);

  localparam cal_WIDTH = (in0_WIDTH > in1_WIDTH) ? in0_WIDTH : in1_WIDTH;

  //------------------------Local signal-------------------
  reg  [in0_WIDTH-1:0] dividend_tmp[  0:in0_WIDTH];
  reg  [in1_WIDTH-1:0] divisor_tmp [  0:in0_WIDTH];
  reg  [in0_WIDTH-1:0] remd_tmp    [  0:in0_WIDTH];
  wire [in0_WIDTH-1:0] comb_tmp    [0:in0_WIDTH-1];
  wire [  cal_WIDTH:0] cal_tmp     [0:in0_WIDTH-1];
  reg  [          1:0] sign_tmp    [  0:in0_WIDTH];
  //------------------------Body---------------------------
  assign quot   = dividend_tmp[in0_WIDTH];
  assign remd   = remd_tmp[in0_WIDTH];
  assign sign_o = sign_tmp[in0_WIDTH];

  // dividend_tmp[0], divisor_tmp[0], remd_tmp[0]
  always @(posedge clk) begin
    if (ce) begin
      dividend_tmp[0] <= dividend;
      divisor_tmp[0]  <= divisor;
      sign_tmp[0]     <= sign_i;
      remd_tmp[0]     <= 1'b0;
    end
  end

  genvar i;
  generate
    for (i = 0; i < in0_WIDTH; i = i + 1) begin : loop
      if (in0_WIDTH == 1) assign comb_tmp[i] = dividend_tmp[i][0];
      else assign comb_tmp[i] = {remd_tmp[i][in0_WIDTH-2:0], dividend_tmp[i][in0_WIDTH-1]};
      assign cal_tmp[i] = {1'b0, comb_tmp[i]} - {1'b0, divisor_tmp[i]};

      always @(posedge clk) begin
        if (ce) begin
          if (in0_WIDTH == 1) dividend_tmp[i+1] <= ~cal_tmp[i][cal_WIDTH];
          else dividend_tmp[i+1] <= {dividend_tmp[i][in0_WIDTH-2:0], ~cal_tmp[i][cal_WIDTH]};
          divisor_tmp[i+1] <= divisor_tmp[i];
          remd_tmp[i+1]    <= cal_tmp[i][cal_WIDTH] ? comb_tmp[i] : cal_tmp[i][in0_WIDTH-1:0];
          sign_tmp[i+1]    <= sign_tmp[i];
        end
      end
    end
  endgenerate

endmodule

module div_sdiv_16ns_16ns_16_20_1_divider
#(parameter
    in0_WIDTH = 32,
    in1_WIDTH = 32,
    out_WIDTH = 32
)
(
    input                       clk,
    input                       reset,
    input                       ce,
    input       [in0_WIDTH-1:0] dividend,
    input       [in1_WIDTH-1:0] divisor,
    input       [1:0]           sign_i,
    output wire [1:0]           sign_o,
    output wire [out_WIDTH-1:0] quot,
    output wire [out_WIDTH-1:0] remd
);

localparam cal_WIDTH = (in0_WIDTH > in1_WIDTH)? in0_WIDTH : in1_WIDTH;

//------------------------Local signal-------------------
reg  [in0_WIDTH-1:0] dividend_tmp[0:in0_WIDTH];
reg  [in1_WIDTH-1:0] divisor_tmp[0:in0_WIDTH];
reg  [in0_WIDTH-1:0] remd_tmp[0:in0_WIDTH];
wire [in0_WIDTH-1:0] comb_tmp[0:in0_WIDTH-1];
wire [cal_WIDTH:0]   cal_tmp[0:in0_WIDTH-1];
reg  [1:0]           sign_tmp[0:in0_WIDTH];
//------------------------Body---------------------------
assign  quot    = dividend_tmp[in0_WIDTH];
assign  remd    = remd_tmp[in0_WIDTH];
assign  sign_o  = sign_tmp[in0_WIDTH];

// dividend_tmp[0], divisor_tmp[0], remd_tmp[0]
always @(posedge clk)
begin
    if (ce) begin
        dividend_tmp[0] <= dividend;
        divisor_tmp[0]  <= divisor;
        sign_tmp[0]     <= sign_i;
        remd_tmp[0]     <= 1'b0;
    end
end

genvar i;
generate 
    for (i = 0; i < in0_WIDTH; i = i + 1)
    begin : loop
        if (in0_WIDTH == 1) assign  comb_tmp[i]     = dividend_tmp[i][0];
        else                assign  comb_tmp[i]     = {remd_tmp[i][in0_WIDTH-2:0], dividend_tmp[i][in0_WIDTH-1]};
        assign  cal_tmp[i]      = {1'b0, comb_tmp[i]} - {1'b0, divisor_tmp[i]};

        always @(posedge clk)
        begin
            if (ce) begin
                if (in0_WIDTH == 1) dividend_tmp[i+1] <= ~cal_tmp[i][cal_WIDTH];
                else                dividend_tmp[i+1] <= {dividend_tmp[i][in0_WIDTH-2:0], ~cal_tmp[i][cal_WIDTH]};
                divisor_tmp[i+1]  <= divisor_tmp[i];
                remd_tmp[i+1]     <= cal_tmp[i][cal_WIDTH]? comb_tmp[i] : cal_tmp[i][in0_WIDTH-1:0];
                sign_tmp[i+1]     <= sign_tmp[i];
            end
        end
    end
endgenerate

endmodule

module div_sdiv_16ns_16ns_16_20_1 
#(parameter
        ID   = 1,
        NUM_STAGE   = 2,
        din0_WIDTH   = 32,
        din1_WIDTH   = 32,
        dout_WIDTH   = 32
)
(
        input                           clk,
        input                           reset,
        input                           ce,
        input           [din0_WIDTH-1:0] din0,
        input           [din1_WIDTH-1:0] din1,
        output          [dout_WIDTH-1:0] dout
);
//------------------------Local signal-------------------
reg     [din0_WIDTH-1:0] dividend0;
reg     [din1_WIDTH-1:0] divisor0;
wire    [din0_WIDTH-1:0] dividend_u;
wire    [din1_WIDTH-1:0] divisor_u;
wire    [dout_WIDTH-1:0] quot_u;
wire    [dout_WIDTH-1:0] remd_u;
reg     [dout_WIDTH-1:0] quot;
reg     [dout_WIDTH-1:0] remd;
wire    [1:0]   sign_i;
wire    [1:0]   sign_o;
//------------------------Instantiation------------------
div_sdiv_16ns_16ns_16_20_1_divider #(
    .in0_WIDTH      ( din0_WIDTH ),
    .in1_WIDTH      ( din1_WIDTH ),
    .out_WIDTH      ( dout_WIDTH )
) div_sdiv_16ns_16ns_16_20_1_divider_u (
    .clk      ( clk ),
    .reset    ( reset ),
    .ce       ( ce ),
    .dividend ( dividend_u ),
    .divisor  ( divisor_u ),
    .sign_i   ( sign_i ),
    .sign_o   ( sign_o ),
    .quot     ( quot_u ),
    .remd     ( remd_u )
);
//------------------------Body---------------------------
assign sign_i     = {dividend0[din0_WIDTH-1] ^ divisor0[din1_WIDTH-1], dividend0[din0_WIDTH-1]};
assign dividend_u = dividend0[din0_WIDTH-1]? ~dividend0[din0_WIDTH-1:0] + 1'b1 :
                                              dividend0[din0_WIDTH-1:0];
assign divisor_u  = divisor0[din1_WIDTH-1]?  ~divisor0[din1_WIDTH-1:0] + 1'b1 :
                                              divisor0[din1_WIDTH-1:0];

always @(posedge clk)
begin
    if (ce) begin
        dividend0 <= din0;
        divisor0  <= din1;
    end
end

always @(posedge clk)
begin
    if (ce) begin
        if (sign_o[1])
            quot <= ~quot_u + 1'b1;
        else
            quot <= quot_u;
    end
end

always @(posedge clk)
begin
    if (ce) begin
        if (sign_o[0])
            remd <= ~remd_u + 1'b1;
        else
            remd <= remd_u;
    end
end

assign dout = quot;

endmodule


