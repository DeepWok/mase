`timescale 1ns / 1ps
module fixed_range_augmentation #(
    parameter WIDTH = 16,
    parameter FRAC_WIDTH = 8,
    parameter SQRT2 = 16'b1011010100000100,
    parameter ISQRT2 = 16'b0101101010000010
) (
    input logic[2*WIDTH-1:0] data_a,    // FORMAT: Q(INT_WIDTH).(FRAC_WIDTH).
    input logic[2*WIDTH-1:0] data_b,    // FORMAT: Q(WIDTH).0.
    output logic[2*WIDTH-1:0] data_out  // FORMAT: Q1.(WIDTH-1).
);

    logic[2*WIDTH-1:0] shift_amount;
    logic[2*WIDTH-1:0] res;

    assign shift_amount = (FRAC_WIDTH > data_b) ? 
                                (
                                    // Checking parity.
                                    (FRAC_WIDTH[0] == data_b[0]) ? 
                                        (FRAC_WIDTH - data_b) >> 1 
                                        :
                                        (FRAC_WIDTH - data_b - 1) >> 1
                                ) 
                                : 
                                (
                                    (FRAC_WIDTH == data_b) ? 
                                        0
                                        :
                                        (
                                            (FRAC_WIDTH[0] == data_b[0]) ? 
                                                (data_b - FRAC_WIDTH) >> 1
                                                :
                                                (data_b - FRAC_WIDTH - 1) >> 1
                                        )
                                );

    assign res = (FRAC_WIDTH > data_b) ? 
                    (
                        (FRAC_WIDTH[0] == data_b[0]) ? 
                            data_a << shift_amount
                            :
                            ((data_a * SQRT2) >> (WIDTH-1) << shift_amount)
                    )
                    :
                    (
                        (FRAC_WIDTH == data_b) ?
                            data_a
                            :
                            (FRAC_WIDTH[0] == data_b[0]) ? 
                                data_a >> shift_amount
                                :
                                ((data_a * ISQRT2) >> (WIDTH-1) >> shift_amount)
                    );

    assign data_out = res >> ((WIDTH-1) - FRAC_WIDTH);
                    
endmodule
