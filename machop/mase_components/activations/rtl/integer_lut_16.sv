`timescale 1ns / 1ps
module integer_lut_16 (
	/* verilator lint_off UNUSEDSIGNAL */
	 /* verilator lint_off SELRANGE */
    input logic [3:0] address,
    output logic [16:0] data_out
);
    // Define the memory array for the LUT
    logic [16:0] lut_memory [0:15];

    // Initialize the LUT memory with e^(address) values
    initial begin
        lut_memory[0]  = 17'b10000000000000000; // e^0 = 1, quantized to 16 bits
        lut_memory[1]  = 17'b00101111000101101; // e^-1 ≈ 0.3679, quantized to 16 bits
        lut_memory[2]  = 17'b00010001010100101; // e^-2 ≈ 0.1353, quantized to 16 bits
        lut_memory[3]  = 17'b00000110010111110; // e^-3 ≈ 0.0497, quantized to 16 bits
        lut_memory[4]  = 17'b00000010010110000; // e^-4 ≈ 0.0183, quantized to 16 bits
        lut_memory[5]  = 17'b00000000110111001; // e^-5 ≈ 0.0067, quantized to 16 bits
        lut_memory[6]  = 17'b00000000010100010; // e^-6 ≈ 0.0025, quantized to 16 bits
        lut_memory[7]  = 17'b00000000000111011; // e^-7 ≈ 0.0009, quantized to 16 bits
        lut_memory[8]  = 17'b00000000000010101; // e^-8 ≈ 0.0003, quantized to 16 bits
        lut_memory[9]  = 17'b00000000000001000; // e^-9 ≈ 0.0001, quantized to 16 bits
        lut_memory[10] = 17'b00000000000000010; // e^-10 ≈ 0.0000, quantized to 16 bits
        lut_memory[11] = 17'b00000000000000001; // e^-11 ≈ 0.0000, quantized to 16 bits
        lut_memory[12] = 17'b00000000000000000; // e^-12 ≈ 0.0000, quantized to 16 bits
        lut_memory[13] = 17'b00000000000000000; // e^-13 ≈ 0.0000, quantized to 16 bits
        lut_memory[14] = 17'b00000000000000000; // e^-14 ≈ 0.0000, quantized to 16 bits
        lut_memory[15] = 17'b00000000000000000; // e^-15 ≈ 0.0000, quantized to 16 bits
    end

    // Assign data_out based on the address
    always_comb begin
        data_out = lut_memory[address];
    end
endmodule
