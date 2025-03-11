`timescale 1ns / 1ps
/*
Module      : optimized_variable_shift
Description :
  optimized version of variable shift.
  if shift_value > 0, this module will implement right shift
  if left shift exceeding output range,
  it will automatically clamp into maximum;
*/
module optimized_right_shift #(
    parameter IN_WIDTH = -1,
    parameter BLOCK_SIZE = -1,
    parameter SHIFT_WIDTH = -1,
    parameter OUT_WIDTH = -1
) (
    input logic [IN_WIDTH - 1:0] data_in[BLOCK_SIZE - 1:0],
    input logic [SHIFT_WIDTH - 1:0] shift_value,
    output logic [OUT_WIDTH - 1:0] data_out[BLOCK_SIZE - 1:0]
);
  localparam SHIFT_DATA_WIDTH = IN_WIDTH + OUT_WIDTH - 1; // The maximum left shift value is out_width - 1

  localparam logic signed [OUT_WIDTH-1:0] MIN_VAL = -(2 ** (OUT_WIDTH - 1));
  localparam logic signed [OUT_WIDTH-1:0] MAX_VAL = (2 ** (OUT_WIDTH - 1)) - 1;

  logic [SHIFT_WIDTH - 1:0] abs_shift_value, real_shift_value;
  logic shift_sign;

  logic [SHIFT_DATA_WIDTH - 1:0] shift_data_list[BLOCK_SIZE - 1:0][SHIFT_DATA_WIDTH -1 : 0];

  logic [OUT_WIDTH - 1:0] clamped_out[BLOCK_SIZE - 1:0];

  enum {
    SHIFT_OUT_RANGE,
    SHIFT_IN_RANGE
  } mode;

  assign shift_sign = shift_value[SHIFT_WIDTH-1];

  assign abs_shift_value = (shift_sign) ? (~shift_value + 1) : shift_value;
  assign real_shift_value = (abs_shift_value < SHIFT_DATA_WIDTH - 1) ? abs_shift_value : SHIFT_DATA_WIDTH - 1;

  // There is several things need to be considered
  always_comb begin
    if ((abs_shift_value >= OUT_WIDTH) && (shift_sign)) mode = SHIFT_OUT_RANGE;
    else mode = SHIFT_IN_RANGE;
  end

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    for (genvar j = 0; j < SHIFT_DATA_WIDTH; j++) begin
      always_comb begin
        shift_data_list[i][j] = (shift_sign) ? $signed(data_in[i]) <<< j :
            $signed(data_in[i]) >>> j;
      end
    end
  end
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    signed_clamp #(
        .IN_WIDTH (SHIFT_DATA_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) data_clamp (
        .in_data (shift_data_list[i][real_shift_value]),
        .out_data(clamped_out[i])
    );
  end

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    always_comb begin
      if (data_in[i] == 0) data_out[i] = 0;
      else
        case (mode)
          SHIFT_OUT_RANGE: data_out[i] = (data_in[i][IN_WIDTH-1]) ? MIN_VAL : MAX_VAL;
          SHIFT_IN_RANGE: data_out[i] = clamped_out[i];
          default: data_out[i] = clamped_out[i];
        endcase
    end
  end
endmodule
