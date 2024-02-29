// Temporary module for modelling an inverse square root with variable
// pipeline delay.

`timescale 1ns/1ps

module temp_inv_sqrt #(
    parameter IN_WIDTH           = 8,
    parameter IN_FRAC_WIDTH      = 8,
    parameter OUT_WIDTH          = 8,
    parameter OUT_FRAC_WIDTH     = 8,
    parameter PIPELINE_CYCLES    = 2
) (
    input  logic                clk,
    input  logic                rst,

    input  logic [IN_WIDTH-1:0] in_data,
    input  logic                in_valid,
    output logic                in_ready,

    output logic [IN_WIDTH-1:0] out_data,
    output logic                out_valid,
    input  logic                out_ready
);

initial begin
    assert (IN_WIDTH == OUT_WIDTH);
end

// "Model" the square root operation
// Output a constant 1/4
assign pipe[0].in_data = 1 << (OUT_FRAC_WIDTH-2);
assign pipe[0].in_valid = in_valid;
assign in_ready = pipe[0].in_ready;

// Make the pipeline
for (genvar i = 0; i < PIPELINE_CYCLES; i++) begin : pipe

    logic [IN_WIDTH-1:0] in_data;
    logic in_valid, in_ready;

    logic [IN_WIDTH-1:0] out_data;
    logic out_valid, out_ready;

    register_slice #(
        .DATA_WIDTH  (IN_WIDTH)
    ) reg_slice_inst (
        .clk         (clk),
        .rst         (rst),
        .in_data     (in_data),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .out_data    (out_data),
        .out_valid   (out_valid),
        .out_ready   (out_ready)
    );
end

// Connect up wires
for (genvar i = 1; i < PIPELINE_CYCLES; i++) begin
    assign pipe[i].in_data = pipe[i-1].out_data;
    assign pipe[i].in_valid = pipe[i-1].out_valid;
    assign pipe[i-1].out_ready = pipe[i].in_ready;
end

assign out_data = pipe[PIPELINE_CYCLES-1].out_data;
assign out_valid = pipe[PIPELINE_CYCLES-1].out_valid;
assign pipe[PIPELINE_CYCLES-1].out_ready = out_ready;

endmodule
