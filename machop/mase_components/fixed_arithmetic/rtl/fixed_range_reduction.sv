`timescale 1ns / 1ps
module fixed_range_reduction #(
    parameter WIDTH = 16
) (
    // x
    input logic[2*WIDTH-1:0] data_a,    // FORMAT: Q(INT_WIDTH).(FRAC_WIDTH).
    // x reduced
    output logic[2*WIDTH-1:0] data_out,  // FORMAT: Q1.(WIDTH-1).
    // msb_index
    output logic[WIDTH-1:0] msb_index
);

    integer i;
    always @* begin
        for(i = WIDTH-1; i >= 0; i = i - 1) begin
            if(data_a[i] == 1) begin
                msb_index = i;
                break;
            end
        end
        // TODO: how to handle when the input is 0? This scenario is not meant
        // to happen.
    end

    //assign msb_index = WIDTH-1;
    assign data_out = data_a << (WIDTH - 1 - msb_index);

endmodule
