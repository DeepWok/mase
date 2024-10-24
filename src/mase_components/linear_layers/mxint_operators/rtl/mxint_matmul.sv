/*
Module      : mxint_matmul
Description : This module does a matrix multiplcation between matrices X & Y.

              The dimensions for the matrix multiplcation are:
              n x m * m x k

              or in MASE naming convention
              a_dim1 x a_dim0 * b_dim1 x bdim_0

              Python equivalent:
              out = np.matmul(X, Y)
*/
`timescale 1ns / 1ps

module mxint_matmul #(
    // Total dimensions
    parameter A_TOTAL_DIM0 = 4,
    parameter A_TOTAL_DIM1 = 4,
    parameter B_TOTAL_DIM0 = 4,
    parameter B_TOTAL_DIM1 = 4,  // must equal A_TOTAL_DIM0

    // Compute dimensions
    parameter A_COMPUTE_DIM0 = 2,
    parameter A_COMPUTE_DIM1 = 2,
    parameter B_COMPUTE_DIM0 = 2,
    parameter B_COMPUTE_DIM1 = 2,  // must equal A_COMPUTE_DIM0

    // Input fixed point widths
    parameter A_MAN_WIDTH = 8,
    parameter A_EXP_WIDTH = 1,
    parameter B_MAN_WIDTH = 8,
    parameter B_EXP_WIDTH = 1,

    // Output fixed point widths
    parameter OUT_MAN_WIDTH = 16,
    parameter OUT_EXP_WIDTH = 2,

    // Output casting/rounding
    parameter OUT_SYMMETRIC = 0,

    // Derived Dimensions (Constants)
    localparam C_TOTAL_DIM0   = B_TOTAL_DIM0,
    localparam C_TOTAL_DIM1   = A_TOTAL_DIM1,
    localparam C_COMPUTE_DIM0 = B_COMPUTE_DIM0,
    localparam C_COMPUTE_DIM1 = A_COMPUTE_DIM1,

    // Derived Depth (Constants)
    localparam A_DEPTH_DIM0 = A_TOTAL_DIM0 / A_COMPUTE_DIM0,
    localparam A_DEPTH_DIM1 = A_TOTAL_DIM1 / A_COMPUTE_DIM1,
    localparam B_DEPTH_DIM0 = B_TOTAL_DIM0 / B_COMPUTE_DIM0,
    localparam B_DEPTH_DIM1 = B_TOTAL_DIM1 / B_COMPUTE_DIM1,
    localparam C_DEPTH_DIM0 = C_TOTAL_DIM0 / C_COMPUTE_DIM0,
    localparam C_DEPTH_DIM1 = C_TOTAL_DIM1 / C_COMPUTE_DIM1
) (
    input logic clk,
    input logic rst,

    // Matix A - row-major order
    input  logic [A_MAN_WIDTH-1:0] ma_data[A_COMPUTE_DIM0*A_COMPUTE_DIM1-1:0],
    input  logic [A_EXP_WIDTH-1:0] ea_data,
    input  logic                   a_valid,
    output logic                   a_ready,

    // Matix B - row-major order
    input  logic [B_MAN_WIDTH-1:0] mb_data[B_COMPUTE_DIM0*B_COMPUTE_DIM1-1:0],
    input  logic [B_EXP_WIDTH-1:0] eb_data,
    input  logic                   b_valid,
    output logic                   b_ready,

    // Matrix C - row-major order
    output logic [OUT_MAN_WIDTH-1:0] mout_data[C_COMPUTE_DIM0*C_COMPUTE_DIM1-1:0],
    output logic [OUT_EXP_WIDTH-1:0] eout_data,
    output logic                     out_valid,
    input  logic                     out_ready
);
  initial begin
    // Check dimension constraint not violated
    assert (A_TOTAL_DIM0 == B_TOTAL_DIM1)
    else $fatal("A_TOTAL_DIM0 must equal B_TOTAL_DIM1!");
    assert (A_COMPUTE_DIM0 == B_COMPUTE_DIM1)
    else $fatal("A_COMPUTE_DIM0 must equal B_COMPUTE_DIM1!");

    // Check compute vs. total divisibility
    assert (A_TOTAL_DIM0 % A_COMPUTE_DIM0 == 0)
    else $fatal("A_DIM0 compute is not divisible!");
    assert (A_TOTAL_DIM1 % A_COMPUTE_DIM1 == 0)
    else $fatal("A_DIM1 compute is not divisible!");
    assert (B_TOTAL_DIM0 % B_COMPUTE_DIM0 == 0)
    else $fatal("B_DIM0 compute is not divisible!");
    assert (B_TOTAL_DIM1 % B_COMPUTE_DIM1 == 0)
    else $fatal("B_DIM1 compute is not divisible!");
  end

  localparam A_FLAT_WIDTH = A_MAN_WIDTH * A_COMPUTE_DIM0 * A_COMPUTE_DIM1;
  localparam B_FLAT_WIDTH = B_MAN_WIDTH * B_COMPUTE_DIM0 * B_COMPUTE_DIM1;
  // A buffer exponent
  logic ma_valid, ma_ready, ea_valid, ea_ready;
  logic [A_EXP_WIDTH - 1:0] buffer_ea_data;
  logic buffer_ea_valid, buffer_ea_ready;
  // A Buffer unflatten out
  logic ma_buffer_out_valid, ma_buffer_out_ready;
  logic [A_MAN_WIDTH-1:0] ma_buffer_out_data[A_COMPUTE_DIM0*A_COMPUTE_DIM1-1:0];
  logic ea_buffer_out_valid, ea_buffer_out_ready;
  logic [ A_EXP_WIDTH-1:0] ea_buffer_out_data;

  // Repeat each submatrix in Matrix A stream B_DEPTH_DIM0 times
  // Only if (B_DEPTH_DIM0 > 1)
  logic [A_FLAT_WIDTH-1:0] ma_data_flat;
  logic [A_FLAT_WIDTH-1:0] ma_buffer_out_data_flat;

  // We need to buffer the B matrix
  // TODO: unless A_DEPTH_DIM1 == 1

  // B buffer exponent
  logic mb_valid, mb_ready, eb_valid, eb_ready;
  logic [B_EXP_WIDTH - 1:0] buffer_eb_data;
  logic buffer_eb_valid, buffer_eb_ready;

  logic [B_FLAT_WIDTH-1:0] mb_data_flat;

  // Buffer outputs
  logic [B_FLAT_WIDTH-1:0] mb_buffer_out_data_flat;
  logic mb_buffer_out_valid, mb_buffer_out_ready;
  logic eb_buffer_out_valid, eb_buffer_out_ready;
  logic [B_EXP_WIDTH-1:0] eb_buffer_out_data;

  // Matrix unflatten output

  logic [B_MAN_WIDTH-1:0] mb_buffer_out_data [B_COMPUTE_DIM0*B_COMPUTE_DIM1-1:0];

  localparam SM_OUT_WIDTH = A_MAN_WIDTH + B_MAN_WIDTH + $clog2(A_COMPUTE_DIM0);
  localparam SM_EXP_WIDTH = (A_EXP_WIDTH > B_EXP_WIDTH) ? A_EXP_WIDTH + 1 : B_EXP_WIDTH + 1;
  logic [SM_OUT_WIDTH-1:0] msm_out_data [C_COMPUTE_DIM0*C_COMPUTE_DIM1];
  logic [SM_EXP_WIDTH-1:0] esm_out_data;
  logic msm_out_valid, msm_out_ready;
  logic sm_out_valid, sm_out_ready;

  logic [C_DEPTH_DIM0-1:0] acc_in_valid;
  logic [C_DEPTH_DIM0-1:0] acc_in_ready;
  logic [C_DEPTH_DIM0-1:0] acc_out_valid;
  logic [C_DEPTH_DIM0-1:0] acc_out_ready;
  localparam MAT_ACC_EXP_WIDTH = SM_EXP_WIDTH;
  localparam MAT_ACC_OUT_WIDTH = SM_OUT_WIDTH + 2 ** SM_EXP_WIDTH + $clog2(B_DEPTH_DIM1);
  logic [MAT_ACC_OUT_WIDTH-1:0] macc_out_data[C_DEPTH_DIM0-1:0][C_COMPUTE_DIM0*C_COMPUTE_DIM1-1:0];
  logic [MAT_ACC_EXP_WIDTH-1:0] eacc_out_data[C_DEPTH_DIM0-1:0];

  logic [MAT_ACC_OUT_WIDTH-1:0] mcast_in_data[C_COMPUTE_DIM0*C_COMPUTE_DIM1-1:0];
  logic [MAT_ACC_EXP_WIDTH-1:0] ecast_in_data;
  logic cast_in_valid, cast_in_ready;



  // -----
  // State
  // -----

  localparam MAT_ACC_PTR_WIDTH = C_DEPTH_DIM0 == 1 ? 1 : $clog2(C_DEPTH_DIM0);
  struct {
    // Points to which matrix accumulator should store the simple_matmul output
    logic [MAT_ACC_PTR_WIDTH-1:0] matrix_acc_ptr;
    // Points at which output accumulator should be connected to the out stream
    logic [MAT_ACC_PTR_WIDTH-1:0] output_acc_ptr;
  }
      self, next_self;
  // -----
  // Logic
  // -----

  generate

    // A matrix Buffers
    // buffer for ea_data
    split_n #(
        .N(2)
    ) a_split_i (
        .data_in_valid (a_valid),
        .data_in_ready (a_ready),
        .data_out_valid({ma_valid, ea_valid}),
        .data_out_ready({ma_ready, ea_ready})
    );

    fifo #(
        .DEPTH(A_TOTAL_DIM0 / A_COMPUTE_DIM0),
        .DATA_WIDTH(A_EXP_WIDTH)
    ) ea_ff_inst (
        .clk(clk),
        .rst(rst),
        .in_data(ea_data),
        .in_valid(ea_valid),
        .in_ready(ea_ready),
        .out_data(buffer_ea_data),
        .out_valid(buffer_ea_valid),
        .out_ready(buffer_ea_ready),
        .empty(),
        .full()
    );
    // buffer for ma_data
    if (B_DEPTH_DIM0 > 1) begin : gen_a_buffer

      matrix_flatten #(
          .DATA_WIDTH(A_MAN_WIDTH),
          .DIM0      (A_COMPUTE_DIM0),
          .DIM1      (A_COMPUTE_DIM1)
      ) weight_buffer_flatten_a (
          .data_in (ma_data),
          .data_out(ma_data_flat)
      );

      single_element_repeat #(
          .DATA_WIDTH(A_EXP_WIDTH),
          // Repeat for number of rows in matrix A
          .REPEAT    (B_DEPTH_DIM0)
      ) ea_input_stream_buffer (
          .clk      (clk),
          .rst      (rst),
          .in_data  (buffer_ea_data),
          .in_valid (buffer_ea_valid),
          .in_ready (buffer_ea_ready),
          .out_data (ea_buffer_out_data),
          .out_valid(ea_buffer_out_valid),
          .out_ready(ea_buffer_out_ready)
      );
      single_element_repeat #(
          .DATA_WIDTH(A_FLAT_WIDTH),
          // Repeat for number of rows in matrix A
          .REPEAT    (B_DEPTH_DIM0)
      ) ma_input_stream_buffer (
          .clk      (clk),
          .rst      (rst),
          .in_data  (ma_data_flat),
          .in_valid (ma_valid),
          .in_ready (ma_ready),
          .out_data (ma_buffer_out_data_flat),
          .out_valid(ma_buffer_out_valid),
          .out_ready(ma_buffer_out_ready)
      );

      matrix_unflatten #(
          .DATA_WIDTH(A_MAN_WIDTH),
          .DIM0      (A_COMPUTE_DIM0),
          .DIM1      (A_COMPUTE_DIM1)
      ) weight_buffer_unflatten_a (
          .data_in (ma_buffer_out_data_flat),
          .data_out(ma_buffer_out_data)
      );

    end else begin : gen_a_reg_slice

      // Add a register stage to cut any combinatoral paths to simple matmul
      unpacked_skid_buffer #(
          .DATA_WIDTH(A_MAN_WIDTH),
          .IN_NUM    (A_COMPUTE_DIM0 * A_COMPUTE_DIM1)
      ) ma_input_stream_reg_slice (
          .clk           (clk),
          .rst           (rst),
          .data_in       (ma_data),
          .data_in_valid (ma_valid),
          .data_in_ready (ma_ready),
          .data_out      (ma_buffer_out_data),
          .data_out_valid(ma_buffer_out_valid),
          .data_out_ready(ma_buffer_out_ready)
      );
      skid_buffer #(
          .DATA_WIDTH(A_EXP_WIDTH)
      ) ea_input_stream_reg_slice (
          .clk           (clk),
          .rst           (rst),
          .data_in       (buffer_ea_data),
          .data_in_valid (buffer_ea_valid),
          .data_in_ready (buffer_ea_ready),
          .data_out      (ea_buffer_out_data),
          .data_out_valid(ea_buffer_out_valid),
          .data_out_ready(ea_buffer_out_ready)
      );
    end

    // B matrix Buffers

    split_n #(
        .N(2)
    ) eb_split_i (
        .data_in_valid (b_valid),
        .data_in_ready (b_ready),
        .data_out_valid({mb_valid, eb_valid}),
        .data_out_ready({mb_ready, eb_ready})
    );
    fifo #(
        .DEPTH(B_TOTAL_DIM1 / B_COMPUTE_DIM1),
        .DATA_WIDTH(B_EXP_WIDTH)
    ) b_ff_inst (
        .clk(clk),
        .rst(rst),
        .in_data(eb_data),
        .in_valid(eb_valid),
        .in_ready(eb_ready),
        .out_data(buffer_eb_data),
        .out_valid(buffer_eb_valid),
        .out_ready(buffer_eb_ready),
        .empty(),
        .full()
    );
    if (A_DEPTH_DIM1 > 1) begin : g_circular_buffer
      logic [B_EXP_WIDTH - 1:0] buffer_eb_data_matching[0:0];
      logic [B_EXP_WIDTH - 1:0] eb_buffer_out_data_matching[0:0];
      assign buffer_eb_data_matching[0] = buffer_eb_data;
      assign eb_buffer_out_data = eb_buffer_out_data_matching[0];
      input_buffer #(
          .DATA_WIDTH (B_EXP_WIDTH),
          .IN_NUM     (1),
          // Repeat for number of rows in matrix A
          .REPEAT     (A_DEPTH_DIM1),
          .BUFFER_SIZE(B_DEPTH_DIM0 * B_DEPTH_DIM1)
      ) input_stream_buffer (
          .clk           (clk),
          .rst           (rst),
          .data_in       (buffer_eb_data_matching),
          .data_in_valid (buffer_eb_valid),
          .data_in_ready (buffer_eb_ready),
          .data_out      (eb_buffer_out_data_matching),
          .data_out_valid(eb_buffer_out_valid),
          .data_out_ready(eb_buffer_out_ready)
      );
      input_buffer #(
          .DATA_WIDTH (B_MAN_WIDTH),
          .IN_NUM     (B_COMPUTE_DIM0 * B_COMPUTE_DIM1),
          .REPEAT     (A_DEPTH_DIM1),
          .BUFFER_SIZE(B_DEPTH_DIM0 * B_DEPTH_DIM1)
      ) weight_buffer (
          .clk,
          .rst,

          // Input streaming port
          .data_in(mb_data),
          .data_in_valid(mb_valid),
          .data_in_ready(mb_ready),

          // Output streaming port
          .data_out(mb_buffer_out_data),
          .data_out_valid(mb_buffer_out_valid),
          .data_out_ready(mb_buffer_out_ready)
      );
    end else begin
      assign mb_buffer_out_data = mb_data;
      assign mb_buffer_out_valid = mb_valid;
      assign mb_ready = mb_buffer_out_ready;
      assign eb_buffer_out_data = buffer_eb_data;
      assign eb_buffer_out_valid = buffer_eb_valid;
      assign buffer_eb_ready = eb_buffer_out_ready;
    end
  endgenerate

  // Feed input A & buffered input B into simple matrix mult

  // Simple matrix multiply block's accumulator width
  // We do not round at simple_matmul level as we want to keep high precision
  // and round ourselves after the output accumulation in this matmul module.

  simple_matmul #(
      .N             (A_COMPUTE_DIM1),
      .M             (A_COMPUTE_DIM0),  // == B_COMPUTE_DIM1
      .K             (B_COMPUTE_DIM0),
      .X_WIDTH       (A_MAN_WIDTH),
      .X_FRAC_WIDTH  (0),
      .Y_WIDTH       (B_MAN_WIDTH),
      .Y_FRAC_WIDTH  (0),
      .OUT_WIDTH     (SM_OUT_WIDTH),
      .OUT_FRAC_WIDTH(0)
  ) simple_matmul_inst (
      .clk      (clk),
      .rst      (rst),
      .x_data   (ma_buffer_out_data),
      .x_valid  (ma_buffer_out_valid),
      .x_ready  (ma_buffer_out_ready),
      .y_data   (mb_buffer_out_data),
      .y_valid  (mb_buffer_out_valid),
      .y_ready  (mb_buffer_out_ready),
      .out_data (msm_out_data),
      .out_valid(msm_out_valid),
      .out_ready(msm_out_ready)
  );

  join_n #(
      .NUM_HANDSHAKES(3)
  ) join_inst (
      .data_in_ready ({ea_buffer_out_ready, eb_buffer_out_ready, msm_out_ready}),
      .data_in_valid ({ea_buffer_out_valid, eb_buffer_out_valid, msm_out_valid}),
      .data_out_valid(sm_out_valid),
      .data_out_ready(sm_out_ready)
  );
  assign esm_out_data = $signed(ea_buffer_out_data) + $signed(eb_buffer_out_data);

  // Direct the result of the simple matmul to the correct matrix_accumulator

  for (genvar i = 0; i < C_DEPTH_DIM0; i++) begin : gen_acc
    mxint_accumulator #(
        .DATA_IN_0_PRECISION_0(SM_OUT_WIDTH),
        .DATA_IN_0_PRECISION_1(SM_EXP_WIDTH),
        .BLOCK_SIZE(C_COMPUTE_DIM0 * C_COMPUTE_DIM1),
        .IN_DEPTH(B_DEPTH_DIM1)
    ) matrix_acc_inst (
        .clk             (clk),
        .rst             (rst),
        .mdata_in_0      (msm_out_data),
        .edata_in_0      (esm_out_data),
        .data_in_0_valid (acc_in_valid[i]),
        .data_in_0_ready (acc_in_ready[i]),
        .mdata_out_0     (macc_out_data[i]),
        .edata_out_0     (eacc_out_data[i]),
        .data_out_0_valid(acc_out_valid[i]),
        .data_out_0_ready(acc_out_ready[i])
    );
  end

  mxint_cast #(
      .IN_MAN_WIDTH(MAT_ACC_OUT_WIDTH),
      .IN_EXP_WIDTH(MAT_ACC_EXP_WIDTH),
      .OUT_MAN_WIDTH(OUT_MAN_WIDTH),
      .OUT_EXP_WIDTH(OUT_EXP_WIDTH),
      .BLOCK_SIZE(C_COMPUTE_DIM0 * C_COMPUTE_DIM1)
  ) cast_i (
      .clk,
      .rst,
      .mdata_in(mcast_in_data),
      .edata_in(ecast_in_data),
      .data_in_valid(cast_in_valid),
      .data_in_ready(cast_in_ready),
      .mdata_out(mout_data),
      .edata_out(eout_data),
      .data_out_valid(out_valid),
      .data_out_ready(out_ready)
  );
  for (genvar i = 0; i < C_DEPTH_DIM0; i++) begin : gen_handshake
    // Change which accumulator the output of simple_matmul goes to
    assign acc_in_valid[i]  = self.matrix_acc_ptr == i ? sm_out_valid : 0;

    // Select which accumulator can output on out stream
    assign acc_out_ready[i] = self.output_acc_ptr == i ? cast_in_ready : 0;
  end

  assign sm_out_ready = acc_in_ready[self.matrix_acc_ptr];


  // Logic to handle accumulator selection & output selection.
  always_comb begin
    next_self = self;

    for (int i = 0; i < C_COMPUTE_DIM0 * C_COMPUTE_DIM1; i++) begin
      mcast_in_data[i] = macc_out_data[self.output_acc_ptr][i];
    end
    ecast_in_data = eacc_out_data[self.output_acc_ptr];
    cast_in_valid = acc_out_valid[self.output_acc_ptr];

    // Change accumulator pointer
    if (sm_out_valid && sm_out_ready) begin
      if (self.matrix_acc_ptr == C_DEPTH_DIM0 - 1) begin
        next_self.matrix_acc_ptr = 0;
      end else begin
        next_self.matrix_acc_ptr += 1;
      end
    end

    // Change output pointer
    if (|acc_out_ready && |acc_out_valid) begin
      if (self.output_acc_ptr == C_DEPTH_DIM0 - 1) begin
        next_self.output_acc_ptr = 0;
      end else begin
        next_self.output_acc_ptr += 1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{default: 0};
    end else begin
      self <= next_self;
    end
  end
endmodule
