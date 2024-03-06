
// NOTE: the current LUT is only designed for 16 bit numbers.
// TODO: parameterise the width.
// TODO: parameterise the LUT size.
`timescale 1ns / 1ps
module fixed_lut #(
    parameter WIDTH = 16,
    parameter LUT_POW = 5
) (
    input logic[LUT_POW-1:0] data_a,    // FORMAT: Q(WIDTH).0.
    output logic[WIDTH-1:0] data_out  // FORMAT: Q1.(WIDTH-1).
);

    always_comb begin
        case(data_a)
            5'd0: data_out =  16'b0111111000011010;
            5'd1: data_out =  16'b0111110001001010;
            5'd2: data_out =  16'b0111101010001100;
            5'd3: data_out =  16'b0111100011100010;
            5'd4: data_out =  16'b0111011101001000;
            5'd5: data_out =  16'b0111010110111110;
            5'd6: data_out =  16'b0111010001000011;
            5'd7: data_out =  16'b0111001011010101;
            5'd8: data_out =  16'b0111000101110101;
            5'd9: data_out =  16'b0111000000100010;
            5'd10: data_out =  16'b0110111011011001;
            5'd11: data_out =  16'b0110110110011100;
            5'd12: data_out =  16'b0110110001101010;
            5'd13: data_out =  16'b0110101101000001;
            5'd14: data_out =  16'b0110101000100001;
            5'd15: data_out =  16'b0110100100001011;
            5'd16: data_out =  16'b0110011111111100;
            5'd17: data_out =  16'b0110011011110110;
            5'd18: data_out =  16'b0110010111110111;
            5'd19: data_out =  16'b0110010100000000;
            5'd20: data_out =  16'b0110010000001111;
            5'd21: data_out =  16'b0110001100100101;
            5'd22: data_out =  16'b0110001001000010;
            5'd23: data_out =  16'b0110000101100100;
            5'd24: data_out =  16'b0110000010001100;
            5'd25: data_out =  16'b0101111110111010;
            5'd26: data_out =  16'b0101111011101101;
            5'd27: data_out =  16'b0101111000100101;
            5'd28: data_out =  16'b0101110101100010;
            5'd29: data_out =  16'b0101110010100011;
            5'd30: data_out =  16'b0101101111101001;
            default: data_out =  16'b0101101100110100;
        endcase
    end

endmodule
