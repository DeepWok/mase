`timescale 1ns / 1ps

module fixed_abs_comparator #(
    parameter IN_WIDTH = 16
) (
    input  [IN_WIDTH-1 : 0] data_in_1,
    input  [IN_WIDTH-1 : 0] data_in_2,
    output [IN_WIDTH-1 : 0] data_out
);
  logic unsigned [IN_WIDTH-1:0] abs_1, abs_2;

  assign abs_1 = ($signed(data_in_1) < 0) ? -$signed(data_in_1) : $signed(data_in_1);
  assign abs_2 = ($signed(data_in_2) < 0) ? -$signed(data_in_2) : $signed(data_in_2);

  assign data_out = (abs_1 > abs_2) ? data_in_1 : data_in_2;

endmodule
