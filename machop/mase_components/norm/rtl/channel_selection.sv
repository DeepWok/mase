`timescale 1ns/1ps

module channel_selection #(
    parameter   NUM_CHANNELS    = 2,
    parameter   NUM_BLOCKS      = 4,
    localparam  MAX_STATE       = NUM_CHANNELS * NUM_BLOCKS-1,
    localparam  STATE_WIDTH     = $clog2(MAX_STATE+1),
    localparam  OUT_WIDTH       = $clog2(NUM_CHANNELS)
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 inc,
    output logic[OUT_WIDTH-1:0]  channel
);

generate

    if(NUM_CHANNELS < 2) begin
        assign channel = 0;

    end else begin
        logic[STATE_WIDTH-1:0] state;

        always_ff @(posedge clk) begin
            if (rst) begin
                state <= 0;
            end else if (inc) begin
                if (state >= MAX_STATE) state <= 0;
                else state <= state + 1;
            end
        end

        assign channel = state[STATE_WIDTH-1:STATE_WIDTH-OUT_WIDTH];

    end

endgenerate


endmodule
