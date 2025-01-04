
`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
module gelu_lut #(
    parameter DATA_IN_0_PRECISION_0  = 16,
    parameter DATA_IN_0_PRECISION_1  = 8,
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [4:0] data_in_0,
    output logic [8:0] data_out_0
);


  always_comb begin
    case (data_in_0)
      4'b0000: data_out_0 = 8'b00000000;
      4'b0001: data_out_0 = 8'b00001010;
      4'b0010: data_out_0 = 8'b00010110;
      4'b0011: data_out_0 = 8'b00100101;
      4'b0100: data_out_0 = 8'b00110110;
      4'b0101: data_out_0 = 8'b01001000;
      4'b0110: data_out_0 = 8'b01011010;
      4'b0111: data_out_0 = 8'b01101100;
      4'b1000: data_out_0 = 8'b11111101;
      4'b1001: data_out_0 = 8'b11111100;
      4'b1010: data_out_0 = 8'b11111010;
      4'b1011: data_out_0 = 8'b11111000;
      4'b1100: data_out_0 = 8'b11110110;
      4'b1101: data_out_0 = 8'b11110101;
      4'b1110: data_out_0 = 8'b11110110;
      4'b1111: data_out_0 = 8'b11111010;
      default: data_out_0 = 8'b0;
    endcase
  end
endmodule
