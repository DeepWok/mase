`timescale 1ns / 1ps
/*
Module      : Mxint cast
Description : MxInt Cast between Layers.
*/
module mxint_cast_try1 #(
    parameter IN_MAN_WIDTH = 1,
    parameter IN_MAN_FRAC_WIDTH = IN_MAN_WIDTH - 1,
    parameter IN_EXP_WIDTH = 1,
    parameter OUT_MAN_WIDTH = 1,
    parameter OUT_EXP_WIDTH = 1,
    parameter ROUND_BITS = 4,
    parameter BLOCK_SIZE = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                     clk,
    input  logic                     rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_MAN_WIDTH-1:0] mdata_in      [BLOCK_SIZE-1:0],
    input  logic [ IN_EXP_WIDTH-1:0] edata_in,
    input  logic                     data_in_valid,
    output logic                     data_in_ready,
    output logic [OUT_MAN_WIDTH-1:0] mdata_out     [BLOCK_SIZE-1:0],
    output logic [OUT_EXP_WIDTH-1:0] edata_out,
    output logic                     data_out_valid,
    input  logic                     data_out_ready
);
  //get max_abs_value of input
  localparam LOG2_WIDTH = $clog2(IN_MAN_WIDTH) + 1;

  localparam LOSSLESSS_EDATA_WIDTH = 
    (LOG2_WIDTH > IN_EXP_WIDTH && LOG2_WIDTH > OUT_EXP_WIDTH) ? LOG2_WIDTH + 2 :
    (IN_EXP_WIDTH > OUT_EXP_WIDTH) ? IN_EXP_WIDTH + 2:
    OUT_EXP_WIDTH + 2;

  localparam SHIFT_WIDTH = (OUT_EXP_WIDTH > IN_EXP_WIDTH) ? OUT_EXP_WIDTH + 1 : IN_EXP_WIDTH + 1;
  localparam SHIFT_DATA_WIDTH = OUT_MAN_WIDTH + 1;

  localparam CAST_WIDTH = OUT_MAN_WIDTH + ROUND_BITS;

  logic [IN_MAN_WIDTH - 1:0] mdata_for_max [BLOCK_SIZE - 1:0];
  logic data_for_max_valid, data_for_max_ready;

  logic [IN_MAN_WIDTH-1:0] mdata_for_out [BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] edata_for_out;
  logic data_for_out_valid, data_for_out_ready;

  // Add register slice after log2_max_abs
  logic [LOG2_WIDTH-1:0] log2_max_value_unreg;
  logic log2_max_value_valid_unreg, log2_max_value_ready_unreg;

  logic [LOG2_WIDTH - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  logic [LOSSLESSS_EDATA_WIDTH - 1:0] edata_out_full;
  logic [SHIFT_WIDTH - 1:0] shift_value;
  logic [IN_EXP_WIDTH + SHIFT_WIDTH - 1:0] merge_shift_edata_unreg;

  logic data_out_join_valid, data_out_join_ready;
  // we dont need to implement full shift here, because we'll clamp in the final.
  // in order to avoid shift loss, we set the shift_data_width = OUT_MAN_WIDTH + 1.

  logic [IN_EXP_WIDTH + SHIFT_WIDTH - 1:0] merge_shift_edata_reg;
  logic [IN_MAN_WIDTH-1:0] mdata_for_out_reg [BLOCK_SIZE-1:0];
  logic [SHIFT_WIDTH-1:0] shift_value_reg;

  logic [IN_EXP_WIDTH + SHIFT_WIDTH - 1:0] merge_shift_edata_reg_1;
  logic [IN_MAN_WIDTH-1:0] mdata_for_out_reg_1 [BLOCK_SIZE-1:0];
  logic data_out_reg_valid_1;
  logic data_out_reg_ready_1;

  logic [CAST_WIDTH-1:0] mdata_for_cast [BLOCK_SIZE-1:0];

  logic [OUT_MAN_WIDTH-1:0] mdata_out_unreg [BLOCK_SIZE-1:0];
  logic [OUT_EXP_WIDTH-1:0] edata_out_unreg;

  logic data_out_reg_valid;
  logic data_out_reg_ready;

  mxint_delay #(
      .DATA_PRECISION_0(IN_MAN_WIDTH),
      .DATA_PRECISION_1(IN_EXP_WIDTH),
      .BLOCK_SIZE(BLOCK_SIZE),
      .DELAY_REG_COUNT($clog2(BLOCK_SIZE) + 1)
  ) mxint_delay_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in(mdata_in),
      .edata_in(edata_in),
      .mdata_out(mdata_for_out),
      .edata_out(edata_for_out)
  );
  log2_max_abs #(
      .IN_SIZE (BLOCK_SIZE),
      .IN_WIDTH(IN_MAN_WIDTH)
  ) max_bas_i (
      .clk,
      .rst,
      .data_in_0(mdata_in),
      .data_in_0_valid(data_in_valid),
      .data_in_0_ready(data_in_ready),
      .data_out_0(log2_max_value),
      .data_out_0_valid(log2_max_value_valid),
      .data_out_0_ready(log2_max_value_ready)
  );

  // get edata_out
  assign edata_out_full = $signed(
      log2_max_value
  ) + $signed(
      edata_for_out
  ) - IN_MAN_FRAC_WIDTH;

  signed_clamp #(
      .IN_WIDTH (LOSSLESSS_EDATA_WIDTH),
      .OUT_WIDTH(OUT_EXP_WIDTH)
  ) exp_clamp (
      .in_data (edata_out_full),
      .out_data(edata_out_unreg)
  );
  
  //get shift_valud
  assign shift_value = $signed(
      edata_out_unreg
  ) - $signed(
      edata_for_out
  ) + IN_MAN_FRAC_WIDTH - (CAST_WIDTH - 1);

  optimized_right_shift #(
      .IN_WIDTH(IN_MAN_WIDTH),
      .SHIFT_WIDTH(SHIFT_WIDTH),
      .OUT_WIDTH(CAST_WIDTH),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) ovshift_inst (
      .data_in(mdata_for_out),
      .shift_value(shift_value),
      .data_out(mdata_for_cast)
  );
    fixed_rounding #(
        .IN_SIZE(BLOCK_SIZE),
        .IN_WIDTH(CAST_WIDTH),
        .IN_FRAC_WIDTH(CAST_WIDTH - 1),
        .OUT_WIDTH(OUT_MAN_WIDTH),
        .OUT_FRAC_WIDTH(OUT_MAN_WIDTH - 1)
    ) fixed_cast_inst (
        .data_in(mdata_for_cast),
        .data_out(mdata_out_unreg)  // Changed to feed into skid buffer
    );

  mxint_register_slice #(
      .DATA_PRECISION_0(OUT_MAN_WIDTH),
      .DATA_PRECISION_1(OUT_EXP_WIDTH),
      .IN_NUM(BLOCK_SIZE)
  ) register_slice_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in(mdata_out_unreg),
      .edata_in(edata_out_unreg),
      .data_in_valid(log2_max_value_valid),
      .data_in_ready(log2_max_value_ready),
      .mdata_out(mdata_out),
      .edata_out(edata_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );



endmodule

module delay_reg #(
    parameter DATA_PRECISION_0 = 1,
    parameter DATA_PRECISION_1 = 1,
    parameter DELAY_REG_COUNT = 1
) (
    input logic clk,
    input logic rst,
    input logic [DATA_PRECISION_0-1:0] data_in,
    output logic [DATA_PRECISION_0-1:0] data_out
);
    logic [DATA_PRECISION_0-1:0] data_delay[DELAY_REG_COUNT-1:0];
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < DELAY_REG_COUNT; i++) begin
                data_delay[i] <= '0;
            end
        end else begin
            data_delay[0] <= data_in;
            for (int i = 0; i < DELAY_REG_COUNT-1; i++) begin
                data_delay[i+1] <= data_delay[i];
            end
        end
    end
    assign data_out = data_delay[DELAY_REG_COUNT-1];
endmodule

module mxint_delay #(
    parameter DATA_PRECISION_0 = 1,
    parameter DATA_PRECISION_1 = 1,
    parameter BLOCK_SIZE = 1,
    parameter DELAY_REG_COUNT = 1
) (
    input logic clk,
    input logic rst,
    input logic [DATA_PRECISION_0-1:0] mdata_in [BLOCK_SIZE-1:0],
    input logic [DATA_PRECISION_1-1:0] edata_in,
    output logic [DATA_PRECISION_0-1:0] mdata_out [BLOCK_SIZE-1:0],
    output logic [DATA_PRECISION_1-1:0] edata_out
);
    logic [DATA_PRECISION_0 * BLOCK_SIZE + DATA_PRECISION_1-1:0] data_in_pack;
    logic [DATA_PRECISION_0 * BLOCK_SIZE + DATA_PRECISION_1-1:0] data_out_pack;
    for (genvar i = 0; i < BLOCK_SIZE; i++) begin   
        assign data_in_pack[DATA_PRECISION_0 * (i+1) - 1:DATA_PRECISION_0 * i] = mdata_in[i];
    end
    assign data_in_pack[DATA_PRECISION_0 * BLOCK_SIZE + DATA_PRECISION_1-1:DATA_PRECISION_0 * BLOCK_SIZE] = edata_in;
    delay_reg #(
        .DATA_PRECISION_0(DATA_PRECISION_0 * BLOCK_SIZE + DATA_PRECISION_1),
        .DATA_PRECISION_1(DATA_PRECISION_1),
        .DELAY_REG_COUNT(DELAY_REG_COUNT)
    ) delay_reg_inst (
        .clk(clk),
        .rst(rst),
        .data_in(data_in_pack),
        .data_out(data_out_pack)
    );
    for (genvar i = 0; i < BLOCK_SIZE; i++) begin
        assign mdata_out[i] = data_out_pack[DATA_PRECISION_0 * (i+1) - 1:DATA_PRECISION_0 * i];
    end
    assign edata_out = data_out_pack[DATA_PRECISION_0 * BLOCK_SIZE + DATA_PRECISION_1-1:DATA_PRECISION_0 * BLOCK_SIZE];
endmodule
