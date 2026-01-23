`timescale 1ns / 1ps
/*
Module      : optimized_variable_shift
Description : optimized version of variable shift.
*/
module optimized_variable_shift #(
    parameter IN_WIDTH = -1,
    parameter BLOCK_SIZE = -1,
    parameter SHIFT_WIDTH = -1,
    parameter OUT_WIDTH = -1
) (
    input logic [IN_WIDTH - 1:0] data_in[BLOCK_SIZE - 1:0],
    input logic [SHIFT_WIDTH - 1:0] shift_value,
    output logic [OUT_WIDTH - 1:0] data_out[BLOCK_SIZE - 1:0]
);
  localparam SHIFT_DATA_WIDTH = OUT_WIDTH + 1;
  logic [SHIFT_WIDTH - 1:0] abs_shift_value, real_shift_value;
  assign abs_shift_value = (shift_value[SHIFT_WIDTH-1]) ? (~shift_value + 1) : shift_value;
  assign real_shift_value = (abs_shift_value < SHIFT_DATA_WIDTH)? abs_shift_value: SHIFT_DATA_WIDTH - 1;
  logic [SHIFT_DATA_WIDTH - 1:0] shift_data[BLOCK_SIZE - 1:0];
  logic [SHIFT_DATA_WIDTH - 1:0] shift_data_list[BLOCK_SIZE - 1:0][SHIFT_DATA_WIDTH -1 : 0];
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    for (genvar j = 0; j < SHIFT_DATA_WIDTH; j++) begin
      always_comb begin
        shift_data_list[i][j] = (shift_value[SHIFT_WIDTH-1]) ? $signed(data_in[i]) <<< j :
            $signed(data_in[i]) >>> j;
      end
    end
    assign shift_data[i] = shift_data_list[i][real_shift_value];
  end
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    signed_clamp #(
        .IN_WIDTH (OUT_WIDTH + 1),
        .OUT_WIDTH(OUT_WIDTH)
    ) data_clamp (
        .in_data (shift_data[i]),
        .out_data(data_out[i])
    );
  end
endmodule
