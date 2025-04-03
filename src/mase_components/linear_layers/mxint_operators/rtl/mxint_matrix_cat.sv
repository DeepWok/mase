`timescale 1ns / 1ps

/*
* This module implements torch.cat([t1, t2], dim=-1)
*
* This module assumes that the streaming is always happening in the last
* direciton, i.e the concatenating direction.
*
* E.g to concatenate the following 2 x 2 matricies:
*
*       1   2       5   6       -->   1   2   5   6
*       3   4       7   8       -->   3   4   7   8
*
* The data is expected to be streamed left to right in blocks. So for a block
* size of 1, this module expects 1 2 3 4, and 5 6 7 8 on each input interface.
* The output would be 1 2 5 6 3 4 7 8, hence concatenating the two matricies.
*
* The limitation of this module is that it requires the PARALLELISM parameters
* of the input and output interfaces to be the same. This mainly a limitation
* of mxint_cast. Can later remove this by adding a flow controller on the
* input of the module. The flow controller would ensure that the concatenation
* internally happens using uniform PARALLELISM. This is not a trivial
*/

module mxint_matrix_cat #(
    //---------------------------------------------------//
    //-------------       Software      -----------------//
    //---------------------------------------------------//

    parameter DATA_IN_0_PRECISION_0 = 1,
    parameter DATA_IN_0_PRECISION_1 = 1,

    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    /* verilator lint_on UNUSEDPARAM */

    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_IN_1_PRECISION_0       = 1,
    parameter DATA_IN_1_PRECISION_1       = 1,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 1,
    parameter DATA_OUT_0_PRECISION_1 = 1,

    //---------------------------------------------------//
    //-------------     Hardware Aliases   --------------//
    //---------------------------------------------------//

    localparam CONST_DIM = DATA_IN_0_PARALLELISM_DIM_0,
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = CONST_DIM,
    /* verilator lint_on UNUSEDPARAM */
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 + DATA_IN_1_TENSOR_SIZE_DIM_1,

    localparam DATA_OUT_0_PARALLELISM_DIM_0 = CONST_DIM,
    localparam DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,

    localparam BLOCK_SIZE   = CONST_DIM * DATA_IN_0_PARALLELISM_DIM_1,
    localparam CONCAT_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,
    localparam CONCAT_DIM_1 = DATA_IN_1_TENSOR_SIZE_DIM_1 / DATA_IN_1_PARALLELISM_DIM_1,

    localparam PARALLELISM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    localparam PARALLELISM_1 = DATA_IN_0_PARALLELISM_DIM_1,

    localparam MWIDTH_IN_0 = DATA_IN_0_PRECISION_0,
    localparam EWIDTH_IN_0 = DATA_IN_0_PRECISION_1,

    localparam MWIDTH_IN_1 = DATA_IN_1_PRECISION_0,
    localparam EWIDTH_IN_1 = DATA_IN_1_PRECISION_1,

    localparam MWIDTH_OUT = DATA_OUT_0_PRECISION_0,
    localparam EWIDTH_OUT = DATA_OUT_0_PRECISION_1,
    localparam FIFO_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_1 > DATA_IN_0_TENSOR_SIZE_DIM_1 ? DATA_IN_0_TENSOR_SIZE_DIM_1 : DATA_IN_0_TENSOR_SIZE_DIM_1
) (
    input wire clk,
    input wire rst,

    input  logic [MWIDTH_IN_0-1:0] mdata_in_0     [BLOCK_SIZE-1:0],
    input  wire  [EWIDTH_IN_0-1:0] edata_in_0,
    input  wire                    data_in_0_valid,
    output logic                   data_in_0_ready,

    input  logic [MWIDTH_IN_1-1:0] mdata_in_1     [BLOCK_SIZE-1:0],
    input  wire  [EWIDTH_IN_1-1:0] edata_in_1,
    input  wire                    data_in_1_valid,
    output logic                   data_in_1_ready,

    output logic [ MWIDTH_OUT-1:0] mdata_out_0     [BLOCK_SIZE-1:0],
    output logic [EWIDTH_IN_0-1:0] edata_out_0,
    output logic                   data_out_0_valid,
    input  logic                   data_out_0_ready
);

  function void driveDataOut(logic [MWIDTH_OUT-1:0] mdata[BLOCK_SIZE-1:0],
                             logic [EWIDTH_OUT-1:0] edata);
    for (int i = 0; i < BLOCK_SIZE; i++) begin
      assign mdata_out_0[i] = mdata[i];
    end

    assign edata_out_0 = edata;

  endfunction

  localparam COUNTER_WIDTH = $clog2(DATA_OUT_0_TENSOR_SIZE_DIM_1) + 1;

  initial begin
    assert (DATA_IN_0_PARALLELISM_DIM_1 == DATA_IN_1_PARALLELISM_DIM_1)
    else $error("PARALLELISM Parameters of matrix_cat should all be equal");
    assert (DATA_IN_1_PARALLELISM_DIM_1 == DATA_OUT_0_PARALLELISM_DIM_1)
    else $error("PARALLELISM Parameters of matrix_cat should all be equal");

    assert (DATA_IN_0_PARALLELISM_DIM_0 == DATA_IN_1_PARALLELISM_DIM_0)
    else $error("PARALLELISM Parameters of matrix_cat should all be equal");
    assert (DATA_IN_1_PARALLELISM_DIM_0 == DATA_OUT_0_PARALLELISM_DIM_0)
    else $error("PARALLELISM Parameters of matrix_cat should all be equal");
  end

  typedef enum integer {
    OUT_0,
    OUT_1
  } matrix_cat_state_enum;

  matrix_cat_state_enum                     state_b;
  matrix_cat_state_enum                     state_r;

  logic                 [COUNTER_WIDTH-1:0] out_cntr_b;
  logic                 [COUNTER_WIDTH-1:0] out_cntr_r;

  logic                 [   MWIDTH_OUT-1:0] mdata_in_0_c    [BLOCK_SIZE-1:0];
  logic                 [   EWIDTH_OUT-1:0] edata_in_0_c;
  logic                                     shift_in_0;
  logic                                     fifo_0_full;

  logic                 [   MWIDTH_OUT-1:0] mdata_in_1_c    [BLOCK_SIZE-1:0];
  logic                 [   EWIDTH_OUT-1:0] edata_in_1_c;
  logic                                     shift_in_1;
  logic                                     fifo_1_full;

  logic                 [   MWIDTH_OUT-1:0] mdata_in_1_fifo [BLOCK_SIZE-1:0];
  logic                 [   EWIDTH_OUT-1:0] edata_in_1_fifo;
  logic                                     fifo_1_valid;
  logic                                     fifo_1_ready;

  logic                 [   MWIDTH_OUT-1:0] mdata_in_0_fifo [BLOCK_SIZE-1:0];
  logic                 [   EWIDTH_OUT-1:0] edata_in_0_fifo;
  logic                                     fifo_0_valid;
  logic                                     fifo_0_ready;

  if ((DATA_OUT_0_PRECISION_0  == DATA_IN_0_PRECISION_0) && (DATA_OUT_0_PRECISION_1 == DATA_IN_0_PRECISION_1))
    begin : no_cast_in_0_gen
    always_comb begin
      for (int i = 0; i < BLOCK_SIZE; i++) begin
        mdata_in_0_c[i] = mdata_in_0[i];
      end
      shift_in_0      = data_in_0_valid;
      data_in_0_ready = fifo_0_full;
      edata_in_0_c    = edata_in_0;
    end
  end else begin : cast_in_0_gen
    mxint_cast #(
        .IN_MAN_WIDTH (MWIDTH_IN_0),
        .IN_EXP_WIDTH (EWIDTH_IN_0),
        .OUT_MAN_WIDTH(MWIDTH_OUT),
        .OUT_EXP_WIDTH(EWIDTH_OUT),
        .BLOCK_SIZE   (BLOCK_SIZE)
    ) cast_I (
        .clk           (clk),
        .rst           (rst),
        .mdata_in      (mdata_in_0),
        .edata_in      (edata_in_0),
        .data_in_valid (data_in_0_valid),
        .data_in_ready (data_in_0_ready),
        .mdata_out     (mdata_in_0_c),
        .edata_out     (edata_in_0_c),
        .data_out_valid(shift_in_0),
        .data_out_ready(fifo_0_full)
    );
  end

  if ((DATA_OUT_0_PRECISION_0  == DATA_IN_1_PRECISION_0) && (DATA_OUT_0_PRECISION_1 == DATA_IN_1_PRECISION_1))
    begin : no_cast_in_1_gen
    always_comb begin
      for (int i = 0; i < BLOCK_SIZE; i++) begin
        mdata_in_1_c[i] = mdata_in_1[i];
      end
      shift_in_1      = data_in_1_valid;
      data_in_1_ready = fifo_1_full;
      edata_in_1_c    = edata_in_1;
    end
  end else begin : cast_in_1_gen
    mxint_cast #(
        .IN_MAN_WIDTH (MWIDTH_IN_1),
        .IN_EXP_WIDTH (EWIDTH_IN_1),
        .OUT_MAN_WIDTH(MWIDTH_OUT),
        .OUT_EXP_WIDTH(EWIDTH_OUT),
        .BLOCK_SIZE   (BLOCK_SIZE)
    ) cast_I (
        .clk           (clk),
        .rst           (rst),
        .mdata_in      (mdata_in_1),
        .edata_in      (edata_in_1),
        .data_in_valid (data_in_1_valid),
        .data_in_ready (data_in_1_ready),
        .mdata_out     (mdata_in_1_c),
        .edata_out     (edata_in_1_c),
        .data_out_valid(shift_in_1),
        .data_out_ready(fifo_1_full)
    );
  end

  /* verilator lint_off PINMISSING */
  unpacked_mx_fifo #(
      .DEPTH    (FIFO_DEPTH),
      .MAN_WIDTH(MWIDTH_OUT),
      .EXP_WIDTH(EWIDTH_OUT),
      .IN_SIZE  (BLOCK_SIZE)
  ) fifo_0_I (
      .clk           (clk),
      .rst           (rst),
      .mdata_in      (mdata_in_0_c),
      .edata_in      (edata_in_0_c),
      .data_in_valid (shift_in_0),
      .data_in_ready (fifo_0_full),
      .mdata_out     (mdata_in_0_fifo),
      .edata_out     (edata_in_0_fifo),
      .data_out_valid(fifo_0_valid),
      .data_out_ready(fifo_0_ready)
  );

  unpacked_mx_fifo #(
      .DEPTH    (FIFO_DEPTH),
      .MAN_WIDTH(MWIDTH_OUT),
      .EXP_WIDTH(EWIDTH_OUT),
      .IN_SIZE  (BLOCK_SIZE)
  ) fifo_1_I (
      .clk           (clk),
      .rst           (rst),
      .mdata_in      (mdata_in_1_c),
      .edata_in      (edata_in_1_c),
      .data_in_valid (shift_in_1),
      .data_in_ready (fifo_1_full),
      .mdata_out     (mdata_in_1_fifo),
      .edata_out     (edata_in_1_fifo),
      .data_out_valid(fifo_1_valid),
      .data_out_ready(fifo_1_ready)
  );
  /* verilator lint_on PINMISSING */

  always_comb begin
    state_b = state_r;
    out_cntr_b = out_cntr_r;
    fifo_0_ready    = fifo_0_valid && data_out_0_ready && (out_cntr_b < CONCAT_DIM_0) && (state_r == OUT_0);
    fifo_1_ready    = fifo_1_valid && data_out_0_ready && (out_cntr_b < CONCAT_DIM_1) && (state_r == OUT_1);

    case (state_r)
      OUT_0: begin
        if (fifo_0_ready) begin
          out_cntr_b = out_cntr_r + 1;
          driveDataOut(mdata_in_0_fifo, edata_in_0_fifo);
        end else if (out_cntr_b >= CONCAT_DIM_0) begin
          out_cntr_b = 0;
          state_b    = OUT_1;
        end
      end
      OUT_1: begin
        if (fifo_1_ready) begin
          out_cntr_b = out_cntr_r + 1;
          driveDataOut(mdata_in_1_fifo, edata_in_1_fifo);
        end else if (out_cntr_b >= CONCAT_DIM_1) begin
          out_cntr_b = 0;
          state_b    = OUT_0;
        end
      end
    endcase
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state_r    <= OUT_0;
      out_cntr_r <= '0;
    end else begin
      state_r    <= state_b;
      out_cntr_r <= out_cntr_b;
    end
  end

  assign data_out_0_valid = ((state_r == OUT_0) && fifo_0_ready) || ((state_r == OUT_1) && fifo_1_ready);

endmodule
