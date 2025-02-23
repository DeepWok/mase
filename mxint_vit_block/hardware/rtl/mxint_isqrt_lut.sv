`timescale 1ns / 1ps
module mxint_isqrt_lut #(
    // This module receives a integer input, and outputs a mxint output
    // Notice, the output man width here should be a fixed point number
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter VARIANCE_MAN_WIDTH = 8,
    parameter OUT_MAN_WIDTH = 8, 
    parameter OUT_MAN_FRAC_WIDTH = 4,
    parameter EXP_WIDTH = 4 // Notice, the exp width here won't give any influence on the final result
) (
    input logic clk,
    input logic rst,
    input logic [IN_WIDTH-1:0] data_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic [OUT_MAN_WIDTH-1:0] mdata_out_0,
    output logic [EXP_WIDTH-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
  localparam SHIFTED_VARIANCE_MAN_WIDTH = VARIANCE_MAN_WIDTH + 1;
  localparam SHIFTED_VARIANCE_MAN_FRAC_WIDTH = VARIANCE_MAN_WIDTH - 1;
  localparam SHIFTED_VARIANCE_EXP_WIDTH = EXP_WIDTH + 1;

  logic [VARIANCE_MAN_WIDTH-1:0] mvariance [0:0];
  logic [EXP_WIDTH-1:0] evariance;

  logic [SHIFTED_VARIANCE_MAN_WIDTH-1:0] shifted_mvariance;
  logic [SHIFTED_VARIANCE_EXP_WIDTH-1:0] shifted_evariance;

  logic [EXP_WIDTH-1:0] evariance_abs;
  logic evariance_is_odd;
  
  mxint_cast #(
      .IN_MAN_WIDTH(IN_WIDTH),
      .IN_MAN_FRAC_WIDTH(IN_FRAC_WIDTH), 
      .IN_EXP_WIDTH(EXP_WIDTH),
      .OUT_MAN_WIDTH(VARIANCE_MAN_WIDTH),
      .OUT_EXP_WIDTH(EXP_WIDTH),
      .BLOCK_SIZE(1)
  ) variance_cast (
      .clk(clk),
      .rst(rst),
      .mdata_in({data_in_0}),
      .edata_in(0),
      .data_in_valid(data_in_0_valid), 
      .data_in_ready(data_in_0_ready),
      .mdata_out(mvariance),
      .edata_out(evariance),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

  assign evariance_abs = evariance[EXP_WIDTH-1] ? -evariance : evariance;


  // Check if exponent is odd
  assign evariance_is_odd = evariance_abs[0];

  // Double mantissa and decrement exponent if odd
  always_comb begin
    if (evariance_is_odd) begin
      shifted_mvariance = mvariance[0] << 1;
      shifted_evariance = $signed(evariance) - 1;
    end
    else begin
      shifted_mvariance = $signed(mvariance[0]);
      shifted_evariance = $signed(evariance);
    end
  end

  isqrt_lut #(
      .DATA_IN_0_PRECISION_0 (SHIFTED_VARIANCE_MAN_WIDTH),
      .DATA_IN_0_PRECISION_1 (SHIFTED_VARIANCE_MAN_FRAC_WIDTH),
      .DATA_OUT_0_PRECISION_0(OUT_MAN_WIDTH),
      .DATA_OUT_0_PRECISION_1(OUT_MAN_FRAC_WIDTH)
  ) exp_map (
      .data_in_0 (shifted_mvariance),
      .data_out_0(mdata_out_0)
  );
  assign edata_out_0 = -$signed(shifted_evariance)>>1;
endmodule