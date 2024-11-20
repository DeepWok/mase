`timescale 1ns / 1ps
module mxint_layernorm #(
    // Dimensions
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,

    // Data widths
    parameter DATA_IN_0_PRECISION_0        = 8,
    parameter DATA_IN_0_PRECISION_1        = 4,
    parameter WEIGHT_PRECISION_0           = 8,
    parameter WEIGHT_PRECISION_1           = 4,
    parameter BIAS_PRECISION_0             = 8,
    parameter BIAS_PRECISION_1             = 4,
    parameter ELEMENTWISE_AFFINE           = 0,

    parameter ISQRT_IN_PRECISION_0         = 8, //PREICISION_0 for ISQRT is integer width
    parameter ISQRT_IN_PRECISION_1         = 8, //PREICISION_1 for ISQRT is integer frac width
    parameter ISQRT_OUT_PRECISION_0        = 8,
    parameter ISQRT_OUT_PRECISION_1        = 4,
    parameter NORM_OUT_PRECISION_0        = 8,
    parameter NORM_OUT_PRECISION_1        = 4,

    parameter BIAS_TENSOR_SIZE_DIM_0       = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_PARALLELISM_DIM_0       = DATA_IN_0_PARALLELISM_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1       = 1,
    parameter BIAS_PARALLELISM_DIM_1       = 1,
    parameter WEIGHT_TENSOR_SIZE_DIM_0     = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter WEIGHT_PARALLELISM_DIM_0     = DATA_IN_0_PARALLELISM_DIM_0,
    parameter WEIGHT_TENSOR_SIZE_DIM_1     = 1,
    parameter WEIGHT_PARALLELISM_DIM_1     = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PRECISION_0       = 8,
    parameter DATA_OUT_0_PRECISION_1       = 4
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input  logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    input  logic [WEIGHT_PRECISION_0-1:0] mweight      [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic [WEIGHT_PRECISION_1-1:0] eweight,
    input  logic                          weight_valid,
    output logic                          weight_ready,

    input  logic [BIAS_PRECISION_0-1:0] mbias      [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0],
    input  logic [BIAS_PRECISION_1-1:0] ebias,
    input  logic                        bias_valid,
    output logic                        bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0 [DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  localparam AFFINE_PRECISION_0 = DATA_OUT_0_PRECISION_0 + WEIGHT_PRECISION_0 + 1;
  localparam AFFINE_PRECISION_1 = DATA_OUT_0_PRECISION_1 + WEIGHT_PRECISION_1;

  logic [NORM_OUT_PRECISION_0 - 1:0] mnorm_out [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];
  logic [DATA_OUT_0_PRECISION_1 - 1:0] enorm_out;
  logic [DATA_IN_0_PARALLELISM_DIM_1 - 1:0] parallel_norm_in_valid, parallel_norm_in_ready;
  logic [DATA_IN_0_PARALLELISM_DIM_1 - 1:0] parallel_norm_out_valid, parallel_norm_out_ready;
  logic norm_out_valid, norm_out_ready;
  logic [AFFINE_PRECISION_0 -1:0] uncast_data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0];
  logic [AFFINE_PRECISION_0 - 1:0] casted_bias[DATA_OUT_0_PARALLELISM_DIM_0-1:0];


  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_1; i++) begin : parallel_dim_1
    assign parallel_norm_in_valid[i] = data_in_0_valid;
    assign parallel_norm_out_ready[i] = norm_out_ready;
    mxint_layernorm_1d #(
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        .DATA_IN_0_MAN_WIDTH(DATA_IN_0_PRECISION_0),
        .DATA_IN_0_EXP_WIDTH(DATA_IN_0_PRECISION_1),
        .ISQRT_IN_MAN_WIDTH(ISQRT_IN_PRECISION_0),
        .ISQRT_IN_MAN_FRAC_WIDTH(ISQRT_IN_PRECISION_1),
        .ISQRT_OUT_MAN_WIDTH(ISQRT_OUT_PRECISION_0),
        .ISQRT_OUT_MAN_FRAC_WIDTH(ISQRT_OUT_PRECISION_1),
        .DATA_OUT_0_MAN_WIDTH(NORM_OUT_PRECISION_0),
        .DATA_OUT_0_MAN_FRAC_WIDTH(NORM_OUT_PRECISION_1)
    ) layer_norm_inst (
        .clk,
        .rst,
        .mdata_in_0(mdata_in_0[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
        .edata_in_0(edata_in_0),
        .data_in_0_valid(parallel_norm_in_valid[i]),
        .data_in_0_ready(parallel_norm_in_ready[i]),
        .mdata_out_0(mnorm_out[i*DATA_IN_0_PARALLELISM_DIM_0 + DATA_IN_0_PARALLELISM_DIM_0 - 1:  i*DATA_IN_0_PARALLELISM_DIM_0]),
        .edata_out_0(),
        .data_out_0_valid(parallel_norm_out_valid[i]),
        .data_out_0_ready(parallel_norm_out_ready[i])
    );
  end
  assign enorm_out = 0;
  assign data_in_0_ready = parallel_norm_in_ready[0];
  assign norm_out_valid  = parallel_norm_out_valid[0];

  if (ELEMENTWISE_AFFINE == 1) begin
    localparam WD_PRECISION_0 = NORM_OUT_PRECISION_0 + WEIGHT_PRECISION_0 + 1;
    localparam WD_MAN_FRAC_WIDTH = NORM_OUT_PRECISION_1 + WEIGHT_PRECISION_0 - 1;
    localparam WD_PRECISION_1 = WEIGHT_PRECISION_1;
    logic [BIAS_PRECISION_0-1:0] mbias_buffered  [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
    logic [BIAS_PRECISION_1-1:0] ebias_buffered;
    logic bias_buffered_valid, bias_buffered_ready;

    logic [WEIGHT_PRECISION_0-1:0] mweight_buffered[DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
    logic [WEIGHT_PRECISION_1-1:0] eweight_buffered;
    logic weight_buffered_ready, weight_buffered_valid;

    logic [WD_PRECISION_0 - 1:0] mwd_out [DATA_IN_0_PARALLELISM_DIM_0 - 1 : 0];
    logic [WD_PRECISION_1 - 1:0] ewd_out;
    logic wd_out_valid, wd_out_ready;

    logic affine_out_ready, affine_out_valid;

    logic [WEIGHT_PRECISION_1- 1:0]shift_value;
    logic [WD_PRECISION_0 - 1:0] casted_bias [DATA_OUT_0_PARALLELISM_DIM_0 - 1:0];

    mxint_circular #(
      .DATA_PRECISION_0(BIAS_PRECISION_0),
      .DATA_PRECISION_1(BIAS_PRECISION_1),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0),
      .REPEAT(DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1),
      .BUFFER_SIZE(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
    ) bias_buffer_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mbias),
        .edata_in(ebias),
        .data_in_valid(bias_valid),
        .data_in_ready(bias_ready),
        .mdata_out(mbias_buffered),
        .edata_out(ebias_buffered),
        .data_out_valid(bias_buffered_valid),
        .data_out_ready(bias_buffered_ready)
    );

    mxint_circular #(
        .DATA_PRECISION_0(WEIGHT_PRECISION_0),
        .DATA_PRECISION_1(WEIGHT_PRECISION_1),
        .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0),
        .REPEAT(DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1),
        .BUFFER_SIZE(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)
    ) weight_buffer_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mweight),
        .edata_in(eweight),
        .data_in_valid(weight_valid),
        .data_in_ready(weight_ready),
        .mdata_out(mweight_buffered),
        .edata_out(eweight_buffered),
        .data_out_valid(weight_buffered_valid),
        .data_out_ready(weight_buffered_ready)
    );

    join2 weight_data_join_inst (
        .data_in_valid ({weight_buffered_valid, norm_out_valid}),
        .data_in_ready ({weight_buffered_ready, norm_out_ready}),
        .data_out_valid(wd_out_valid),
        .data_out_ready(wd_out_ready)
    );

    for(genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : affine_bias_parallel
        for(genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : affine_bias_parallel
            localparam int k = i * DATA_IN_0_PARALLELISM_DIM_0 + j;
            assign mwd_out[k] = $signed(mweight_buffered[j]) * $signed(mnorm_out[k]);
        end
    end
    assign ewd_out = $signed(eweight_buffered) + $signed(enorm_out);

    join2 wd_bias_join_inst (
        .data_in_valid ({wd_out_valid, bias_buffered_valid}),
        .data_in_ready ({wd_out_ready, bias_buffered_ready}),
        .data_out_valid(affine_out_valid),
        .data_out_ready(affine_out_ready)
    );
    assign shift_value = $signed(ewd_out) - $signed(ebias_buffered);
    optimized_right_shift #(
        .IN_WIDTH(BIAS_PRECISION_0),
        .SHIFT_WIDTH(WEIGHT_PRECISION_1),
        .OUT_WIDTH(WD_PRECISION_0),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_0)
    ) ovshift_inst (
        .data_in(mbias_buffered),
        .shift_value(shift_value),
        .data_out(casted_bias)
    );
    logic [WD_PRECISION_0 - 1:0] maffine_out [DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_0 - 1:0];
    logic eaffine_out;
    for(genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : affine_bias_parallel_dim_1
        for(genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : affine_bias_parallel_dim_0
            localparam int k = i * DATA_IN_0_PARALLELISM_DIM_0 + j;
            assign maffine_out[k] = $signed(casted_bias[j]) + $signed(mwd_out[k]);
        end
    end
    assign eaffine_out = ewd_out;

    mxint_cast #(
        .IN_MAN_WIDTH(WD_PRECISION_0),
        .IN_MAN_FRAC_WIDTH(WD_MAN_FRAC_WIDTH),
        .IN_EXP_WIDTH(WD_PRECISION_0),
        .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
    ) u_mxint_cast (
        .clk(clk),
        .rst(rst),
        .mdata_in(maffine_out),
        .edata_in(eaffine_out),
        .data_in_valid(affine_out_valid),
        .data_in_ready(affine_out_ready),
        .mdata_out(mdata_out_0),
        .edata_out(edata_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
  end else begin
    mxint_cast #(
        .IN_MAN_WIDTH(NORM_OUT_PRECISION_0),
        .IN_MAN_FRAC_WIDTH(NORM_OUT_PRECISION_1),
        .IN_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
        .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
        .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
        .BLOCK_SIZE(DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_0)
    ) u_mxint_cast (
        .clk(clk),
        .rst(rst),
        .mdata_in(mnorm_out),
        .edata_in(enorm_out),
        .data_in_valid(norm_out_valid),
        .data_in_ready(norm_out_ready),
        .mdata_out(mdata_out_0),
        .edata_out(edata_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );
  end

endmodule

module dim_0_cast #(
    parameter MAN_WIDTH = 8,
    parameter EXP_WIDTH = 4,
    parameter IN_DEPTH = 10,
    parameter BLOCK_SIZE = 4
) (
    input logic clk,
    input logic rst,
    input logic data_in_0_valid,
    output logic data_in_0_ready,
    input logic [MAN_WIDTH-1:0] mdata_in_0[BLOCK_SIZE-1:0],
    input logic [EXP_WIDTH-1:0] edata_in_0,

    output logic data_out_0_valid,
    input logic data_out_0_ready,
    output logic [MAN_WIDTH-1:0] mdata_out_0[BLOCK_SIZE-1:0],
    output logic [EXP_WIDTH-1:0] edata_out_0
); 

    // Internal signals
    logic [MAN_WIDTH-1:0] mdata_in_0_fifo[BLOCK_SIZE-1:0];
    logic [EXP_WIDTH-1:0] edata_in_0_fifo;
    logic data_in_0_fifo_valid;
    logic data_in_0_fifo_ready;
    
    logic [EXP_WIDTH-1:0] edata_in_0_straight;
    logic data_in_0_straight_valid;
    logic data_in_0_straight_ready;
    
    logic [EXP_WIDTH-1:0] max_edata_in_0;
    logic max_edata_in_0_valid;
    logic max_edata_in_0_ready;
    
    logic [EXP_WIDTH-1:0] circular_max_edata_in_0;
    logic circular_max_edata_in_0_valid;
    logic circular_max_edata_in_0_ready;

    logic signed [EXP_WIDTH:0] shift_value;

    // Split2 circuit for parallel processing
    unpacked_mx_split2_with_data #(
        .DEPTH(IN_DEPTH),
        .MAN_WIDTH(MAN_WIDTH),
        .EXP_WIDTH(EXP_WIDTH),
        .IN_SIZE(BLOCK_SIZE)
    ) split2_circ (
        .clk(clk),
        .rst(rst),
        // Input from circular buffer
        .mdata_in(mdata_in_0),
        .edata_in(edata_in_0),
        .data_in_valid(data_in_0_valid),
        .data_in_ready(data_in_0_ready),
        // FIFO output path (not used)
        .fifo_mdata_out(mdata_in_0_fifo),
        .fifo_edata_out(edata_in_0_fifo),
        .fifo_data_out_valid(data_in_0_fifo_valid),
        .fifo_data_out_ready(data_in_0_fifo_ready),
        // Straight output path
        .straight_mdata_out(),  // Connect to the same signals previously used
        .straight_edata_out(edata_in_0_straight),
        .straight_data_out_valid(data_in_0_straight_valid),
        .straight_data_out_ready(data_in_0_straight_ready)
    );

    // Sequential max finder
    sequential_max #(
        .IN_DEPTH(IN_DEPTH),
        .IN_WIDTH(EXP_WIDTH)
    ) sequential_max_inst (
        .clk            (clk),             // input
        .rst            (rst),             // input
        .data_in        (edata_in_0_straight),         // input  [IN_WIDTH-1:0]
        .data_in_valid  (data_in_0_straight_valid),   // input
        .data_in_ready  (data_in_0_straight_ready),   // output
        .data_out       (max_edata_in_0),        // output [IN_WIDTH-1:0]
        .data_out_valid (max_edata_in_0_valid),  // output
        .data_out_ready (max_edata_in_0_ready)   // input
    );
  input_buffer #(
      .DATA_WIDTH (EXP_WIDTH),
      .IN_NUM     (1),
      .REPEAT     (IN_DEPTH),
      .BUFFER_SIZE(1)
  ) mdata_in_0_buffer (
      .clk,
      .rst,
      // Input streaming port
      .data_in({max_edata_in_0}),
      .data_in_valid(max_edata_in_0_valid),
      .data_in_ready(max_edata_in_0_ready),
      // Output streaming port
      .data_out({circular_max_edata_in_0}),
      .data_out_valid(circular_max_edata_in_0_valid),
      .data_out_ready(circular_max_edata_in_0_ready)
  );

    // Join circuit for output synchronization
    join2 data_out_join_inst (
        .data_in_ready({circular_max_edata_in_0_ready, data_in_0_fifo_ready}),
        .data_in_valid({circular_max_edata_in_0_valid, data_in_0_fifo_valid}),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );

    // Calculate shift value and perform optimized right shift
    assign shift_value = $signed(max_edata_in_0) - $signed(circular_max_edata_in_0);

    optimized_right_shift #(
        .IN_WIDTH(MAN_WIDTH),
        .SHIFT_WIDTH(EXP_WIDTH),
        .OUT_WIDTH(MAN_WIDTH),
        .BLOCK_SIZE(BLOCK_SIZE)
    ) ovshift_inst (
        .data_in(mdata_in_0_fifo),
        .shift_value(shift_value),
        .data_out(mdata_out_0)
    );

    // Assign final exponent output
    assign edata_out_0 = max_edata_in_0;

endmodule

module mxint_layernorm_1d #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    parameter DATA_IN_0_MAN_WIDTH = 8,
    parameter DATA_IN_0_MAN_FRAC_WIDTH = 8,
    parameter DATA_IN_0_EXP_WIDTH = 4,

    parameter DATA_OUT_0_MAN_WIDTH = 8,
    parameter DATA_OUT_0_MAN_FRAC_WIDTH = 8,
    parameter DATA_OUT_0_EXP_WIDTH = 4,

    parameter ISQRT_IN_MAN_WIDTH = 8,
    parameter ISQRT_IN_MAN_FRAC_WIDTH = 4,
    parameter ISQRT_OUT_MAN_WIDTH = 8,
    parameter ISQRT_OUT_MAN_FRAC_WIDTH = 4
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input clk,
    input rst,

    input logic data_in_0_valid,
    output logic data_in_0_ready,
    input logic [DATA_IN_0_MAN_WIDTH-1:0] mdata_in_0[DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    input logic [DATA_IN_0_EXP_WIDTH-1:0] edata_in_0,

    output logic data_out_0_valid,
    input logic data_out_0_ready,
    output logic [DATA_OUT_0_MAN_WIDTH-1:0] mdata_out_0[DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_EXP_WIDTH-1:0] edata_out_0
);
    // Internal signals
    logic [DATA_IN_0_MAN_WIDTH-1:0] casted_mdata_in[DATA_IN_0_PARALLELISM_DIM_0-1:0];
    logic [DATA_IN_0_EXP_WIDTH-1:0] casted_edata_in;
    logic casted_data_in_valid;
    logic casted_data_in_ready;
    
    dim_0_cast #(
        .MAN_WIDTH(DATA_IN_0_MAN_WIDTH),
        .EXP_WIDTH(DATA_IN_0_EXP_WIDTH),
        .IN_DEPTH(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0),
        .BLOCK_SIZE(DATA_IN_0_PARALLELISM_DIM_0)
    ) u_dim_0_cast (
        .clk(clk),
        .rst(rst),
        .data_in_0_valid(data_in_0_valid),
        .data_in_0_ready(data_in_0_ready),
        .mdata_in_0(mdata_in_0),
        .edata_in_0(edata_in_0),
        .data_out_0_valid(casted_data_in_valid),
        .data_out_0_ready(casted_data_in_ready),
        .mdata_out_0(casted_mdata_in),
        .edata_out_0(casted_edata_in)
    );

    layer_norm_1d #(
        .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        // Data widths
        .DATA_IN_0_PRECISION_0(DATA_IN_0_MAN_WIDTH),
        .DATA_IN_0_PRECISION_1(DATA_IN_0_MAN_FRAC_WIDTH),
        .ISQRT_IN_PRECISION_0(ISQRT_IN_MAN_WIDTH),
        .ISQRT_IN_PRECISION_1(ISQRT_IN_MAN_FRAC_WIDTH),
        .ISQRT_OUT_PRECISION_0(ISQRT_OUT_MAN_WIDTH),
        .ISQRT_OUT_PRECISION_1(ISQRT_OUT_MAN_FRAC_WIDTH),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0),
        .DATA_OUT_0_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0),
        .DATA_OUT_0_PRECISION_0(DATA_OUT_0_MAN_WIDTH),
        .DATA_OUT_0_PRECISION_1(DATA_OUT_0_MAN_FRAC_WIDTH)
    ) u_layer_norm_1d (
        .clk(clk),
        .rst(rst),
        .data_in_0(casted_mdata_in),
        .data_in_0_valid(casted_data_in_valid),
        .data_in_0_ready(casted_data_in_ready),
        .data_out_0(mdata_out_0),
        .data_out_0_valid(data_out_0_valid),
        .data_out_0_ready(data_out_0_ready)
    );

    assign edata_out_0 = 0;

endmodule
