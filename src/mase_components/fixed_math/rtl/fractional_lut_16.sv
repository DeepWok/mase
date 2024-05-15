`timescale 1ns / 1ps
module fractional_lut_16 (
    /* verilator lint_off UNUSEDSIGNAL */
    /* verilator lint_off SELRANGE */
    input  logic [ 2:0] address,
    output logic [16:0] data_out
);
  // Define the memory array for the LUT
  logic [16:0] lut_memory[0:7];

  // Values of e^(-Address * 2^(-3)) quantized to 16 bits
  initial begin
    lut_memory[0] = 17'b10000000000000000;  // e^(-0.0) ≈ 1, quantized to 16 bits
    lut_memory[1] = 17'b01110000111101011;  // e^(-0.875) ≈ 0.8828, quantized to 16 bits
    lut_memory[2] = 17'b01100011101011111;  // e^(-0.750) ≈ 0.7783, quantized to 16 bits
    lut_memory[3] = 17'b01010111111110010;  // e^(-0.625) ≈ 0.6919, quantized to 16 bits
    lut_memory[4] = 17'b01001101101000101;  // e^(-0.500) ≈ 0.6102, quantized to 16 bits
    lut_memory[5] = 17'b01000100100000110;  // e^(-0.375) ≈ 0.5413, quantized to 16 bits
    lut_memory[6] = 17'b00111100011101101;  // e^(-0.250 ) ≈ 0.4835, quantized to 16 bits
    lut_memory[7] = 17'b00110101010110111;  // e^(-0.125 ) ≈ 0.4346, quantized to 16 bits
  end

  // Assign data_out based on the address
  always_comb begin
    data_out = lut_memory[address];
  end
endmodule
