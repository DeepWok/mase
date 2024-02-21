/*
Module      : group_norm_2d
Description : This module calculates the generalised group norm.
              https://arxiv.org/abs/1803.08494v3

              This module can be easily trivially specialised into layer norm or
              instance norm by setting the GROUP_CHANNELS param to equal C or 1
              respectively.

              Group norm is independent of batch size, so the input shape is:
              (GROUP, DEPTH_DIM1 * DEPTH_DIM0, COMPUTE_DIM1 * COMPUTE_DIM0)
*/

`timescale 1ns/1ps
`default_nettype none

module group_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 6,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter GROUP_CHANNELS      = 2,

    // Data widths
    parameter IN_WIDTH            = 8,
    parameter IN_FRAC_WIDTH       = 2,
    parameter OUT_WIDTH           = 8,
    parameter OUT_FRAC_WIDTH      = 7
) (
    input  logic             clk,
    input  logic             rst,

    input  logic [IN_WIDTH-1:0] in_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic             in_valid,
    output logic             in_ready,

    output logic [IN_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic             out_valid,
    input  logic             out_ready
);

// Constant derived params
localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;

localparam NUM_ITERS = DEPTH_DIM0 * DEPTH_DIM1 * GROUP_CHANNELS;
localparam ITER_WIDTH = $clog2(NUM_ITERS);

// State
struct {
    // Batched Difference: X - mu
    logic [IN_WIDTH-1:0] diff [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic diff_valid;

    // Batched Squared: (X - mu)^2
    logic [(IN_WIDTH*2)-1:0] squared [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic squared_valid;

    // Batched Variance: (X - mu)^2 / N
    // TODO: change this width to more precise??
    logic [(IN_WIDTH*2)-1:0] variance [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic variance_valid;

} self, next_self;


// Input FIFO
localparam DATA_FLAT_WIDTH = IN_WIDTH * COMPUTE_DIM0 * COMPUTE_DIM1;
// localparam FIFO_DEPTH = GROUP_CHANNELS * DEPTH_DIM0 * DEPTH_DIM1;

logic [DATA_FLAT_WIDTH-1:0] in_data_flat, out_data_flat;
logic [IN_WIDTH-1:0] fifo_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic fifo_out_valid, fifo_out_ready;
logic fifo_in_valid, fifo_in_ready;

matrix_flatten #(
    .DATA_WIDTH(IN_WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) input_flatten (
    .data_in(in_data),
    .data_out(in_data_flat)
);

fifo_v2 #(
    .SIZE(NUM_ITERS),
    .DATA_WIDTH(DATA_FLAT_WIDTH)
) fifo_inst (
    .clk(clk),
    .rst(rst),
    .in_data(in_data_flat),
    .in_valid(fifo_in_valid),
    .in_ready(fifo_in_ready),
    .out_data(out_data_flat),
    .out_valid(fifo_out_valid),
    .out_ready(fifo_out_ready)
);

matrix_unflatten #(
    .DATA_WIDTH(IN_WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) fifo_unflatten (
    .data_in(out_data_flat),
    .data_out(fifo_data)
);

// Input Adder Tree
localparam ADDER_TREE_IN_SIZE = COMPUTE_DIM0 * COMPUTE_DIM1;
localparam ADDER_TREE_OUT_WIDTH = $clog2(ADDER_TREE_IN_SIZE) + IN_WIDTH;

logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_data;
logic adder_tree_out_valid, adder_tree_out_ready;
logic adder_tree_in_valid, adder_tree_in_ready;

fixed_adder_tree #(
    .IN_SIZE(COMPUTE_DIM0 * COMPUTE_DIM1),
    .IN_WIDTH(IN_WIDTH),
) sum_adder_tree (
    .clk(clk),
    .rst(rst),
    .data_in(in_data),
    .data_in_valid(adder_tree_in_valid),
    .data_in_ready(adder_tree_in_ready),
    .data_out(adder_tree_data),
    .data_out_valid(adder_tree_out_valid),
    .data_out_ready(adder_tree_out_ready)
);

// Split2 for input to FIFO & Adder Tree
split2 input_fifo_adder_split (
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out_valid({adder_tree_in_valid, fifo_in_valid}),
    .data_out_ready({adder_tree_in_ready, fifo_in_ready})
);


// Accumulator for mu
// localparam ACC_DETPH = GROUP_CHANNELS * DEPTH_DIM0 * DEPTH_DIM1;
localparam ACC_OUT_WIDTH = $clog2(NUM_ITERS) + ADDER_TREE_OUT_WIDTH;

logic [ACC_OUT_WIDTH-1:0] mu_acc, mu_acc_div;
logic mu_acc_valid, mu_acc_ready;

fixed_accumulator #(
    .IN_DEPTH(NUM_ITERS),
    .IN_WIDTH(ADDER_TREE_OUT_WIDTH)
) mu_accumulator (
    .clk(clk),
    .rst(rst),
    .data_in(adder_tree_data),
    .data_in_valid(adder_tree_out_valid),
    .data_in_ready(adder_tree_out_ready),
    .data_out(mu_acc),
    .data_out_valid(mu_acc_valid),
    .data_out_ready(mu_acc_ready)
);

// Division logic for mu
logic [IN_WIDTH-1:0] mu_in, mu_out;
logic mu_out_valid, mu_out_ready;

assign mu_acc_div = mu_acc / NUM_ITERS;
assign mu_in = mu_acc_div[IN_WIDTH-1:0];

repeat_circular_buffer #(
    .DATA_WIDTH(IN_WIDTH),
    .REPEAT(NUM_ITERS),
    .SIZE(1)
) mu_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(mu_in),
    .in_valid(mu_acc_valid),
    .in_ready(mu_acc_ready),
    .out_data(mu_out),
    .out_valid(mu_out_valid),
    .out_ready(mu_out_ready)
);

// Join 2 for combining fifo and mu buffer signals
logic mu_fifo_valid, mu_fifo_ready;

join2 mu_fifo_join2 (
    .data_in_valid({mu_out_valid, fifo_out_valid}),
    .data_in_ready({mu_out_ready, fifo_out_ready}),
    .data_out_valid(mu_fifo_valid),
    .data_out_ready(mu_fifo_ready)
);

// Subtract -> Square -> Divide Pipeline
for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : var_pipeline

    logic [IN_WIDTH-1:0] diff_in, diff_out;
    logic diff_out_valid, diff_out_ready;
    assign diff_in = fifo_data[i] - mu_out;

    skid_buffer #(
        .DATA_WIDTH(IN_WIDTH)
    ) subtract_ff (
        .clk(clk),
        .rst(rst),
        .data_in(diff_in),
        .data_in_valid(mu_fifo_valid),
        .data_in_ready(mu_fifo_ready),
        .data_out(diff_out),
        .data_out_valid(diff_out_valid),
        .data_out_ready(diff_out_ready)
    );

    // Assign the output of diff int batch to be buffered
    assign diff_batch_in[i] = diff_out;

    logic [(IN_WIDTH*2)-1:0] square_in, square_out;
    logic square_out_valid, square_out_ready;
    assign square_in = diff_out * diff_out;

    skid_buffer #(
        .DATA_WIDTH(IN_WIDTH*2)
    ) square_ff (
        .clk(clk),
        .rst(rst),
        .data_in(square_in),
        .data_in_valid(diff_out_valid),
        .data_in_ready(diff_out_ready),
        .data_out(square_out),
        .data_out_valid(square_out_valid),
        .data_out_ready(square_out_ready)
    );

    logic [(IN_WIDTH*2)-1:0] variance_in, variance_out;
    logic variance_out_valid, variance_out_ready;
    assign variance_in = square_out / NUM_ITERS;

    skid_buffer #(
        .DATA_WIDTH(IN_WIDTH*2)
    ) variance_ff (
        .clk(clk),
        .rst(rst),
        .data_in(variance_in),
        .data_in_valid(square_out_valid),
        .data_in_ready(square_out_ready),
        .data_out(variance_out),
        .data_out_valid(variance_out_valid),
        .data_out_ready(variance_out_ready)
    );

end

// FIFO for differences
logic [IN_WIDTH-1:0] diff_batch_in [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic [IN_WIDTH-1:0] diff_batch_out [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic [IN_WIDTH*COMPUTE_DIM0*COMPUTE_DIM1-1:0] diff_batch_in_flat;
logic [IN_WIDTH*COMPUTE_DIM0*COMPUTE_DIM1-1:0] diff_batch_out_flat;

matrix_flatten #(
    .DATA_WIDTH(IN_WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) input_flatten (
    .data_in(diff_batch),
    .data_out(diff_batch_in_flat)
);

fifo_v2 #(
    .SIZE(),
    .DATA_WIDTH(DATA_FLAT_WIDTH)
) fifo_inst (
    .clk(clk),
    .rst(rst),
    .in_data(diff_batch_in_flat),
    .in_valid(),
    .in_ready(),
    .out_data(diff_batch_out_flat),
    .out_valid(),
    .out_ready()
);

matrix_unflatten #(
    .DATA_WIDTH(IN_WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) fifo_unflatten (
    .data_in(diff_batch_out_flat),
    .data_out(diff_batch_out)
);


always_comb begin
    next_self = self;

    // // mu_acc_ready = !self.mu_valid;
    // // fifo_out_ready = self.mu_valid && fifo_out_valid;

    // // // Division logic to get mu
    // // if (mu_acc_valid && mu_acc_ready) begin
    // //     // Truncation rounding
    // //     mu_acc_div = mu_acc / (GROUP_CHANNELS * DEPTH_DIM0 * DEPTH_DIM1);
    // //     next_self.mu = mu_acc_div[IN_WIDTH-1:0];
    // //     next_self.mu_valid = 1;
    // //     next_self.mu_counter = 0;
    // // end

    // mu_fifo_ready = !self.diff_valid; // More complex logic to ff

    // // Batched subtract logic
    // if (mu_fifo_valid && mu_fifo_ready) begin
    //     for (int i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin
    //         next_self.diff[i] = fifo_data[i] - mu_out;
    //     end
    //     next_self.diff_valid = 1;
    // end

    // // Batched squared logic
    // if (self.diff_valid && !self.squared_valid) begin
    //     for (int i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin
    //         next_self.squared[i] = $signed(self.diff[i]) * $signed(self.diff[i]);
    //     end
    //     next_self.squared_valid = 1;
    // end

    // // Batched variance logic
    // if (self.squared_valid && !self.variance_valid) begin
    //     for (int i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin
    //         logic [(IN_WIDTH*2)-1:0] squared_div;
    //         squared_div = self.squared[i] / NUM_ITERS;
    //         next_self.variance[i] = squared_div[IN_WIDTH-1:0];
    //     end
    //     next_self.variance_valid = 1;
    // end

end

always_ff @(posedge clk) begin
    if (rst) begin
        self <= '{default: '0};
    end else begin
        self <= next_self;
    end
end

endmodule
