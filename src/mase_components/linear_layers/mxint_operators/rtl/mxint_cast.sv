`timescale 1ns / 1ps

/*
Module      : MxInt Cast
Description : MxInt Cast between Layers. The input layer does not have to be normalized.
              The output layer will be normalized.
*/

module mxint_cast #(
    parameter IN_MAN_WIDTH = 1,
    parameter IN_EXP_WIDTH = 1,
    parameter OUT_MAN_WIDTH = 1,
    parameter OUT_EXP_WIDTH = 1,
    parameter BLOCK_SIZE = 1
) (
    input logic clk,
    input logic rst,

    // Input Data
    input  logic signed [IN_MAN_WIDTH-1:0] mdata_in     [BLOCK_SIZE-1:0],
    input  logic        [IN_EXP_WIDTH-1:0] edata_in,
    input  logic                           data_in_valid,
    output logic                           data_in_ready,

    // Output Data
    output logic signed [OUT_MAN_WIDTH-1:0] mdata_out     [BLOCK_SIZE-1:0],
    output logic        [OUT_EXP_WIDTH-1:0] edata_out,
    output logic                            data_out_valid,
    input  logic                            data_out_ready
);


  // =============================
  // Internal Signals
  // =============================

  logic data_for_max_valid, data_for_max_ready, data_for_out_valid, data_for_out_ready;
  logic [IN_MAN_WIDTH-1:0] mbuffer_data_for_out [BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] ebuffer_data_for_out;
  logic buffer_data_for_out_valid, buffer_data_for_out_ready;

  localparam LOG2_WIDTH = $clog2(IN_MAN_WIDTH) + 1;
  logic [LOG2_WIDTH - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  localparam EBIAS_OUT = 2 ** (OUT_EXP_WIDTH - 1) - 1;
  localparam EBIAS_IN = 2 ** (IN_EXP_WIDTH - 1) - 1;
  localparam LOSSLESSS_EDATA_WIDTH = max(LOG2_WIDTH, IN_EXP_WIDTH, OUT_EXP_WIDTH) + 2;
  localparam FIFO_DEPTH = $clog2(BLOCK_SIZE);
  logic [LOSSLESSS_EDATA_WIDTH - 1:0] edata_out_full;


  // =============================
  // Handshake Signals
  // =============================

  split2 split_i (
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out_valid({data_for_max_valid, data_for_out_valid}),
      .data_out_ready({data_for_max_ready, data_for_out_ready})
  );

  // =============================
  // Compute Log2 Max Value
  // =============================

  log2_max_abs #(
      .IN_SIZE (BLOCK_SIZE),
      .IN_WIDTH(IN_MAN_WIDTH)
  ) max_bas_i (
      .clk(clk),
      .rst(rst),
      .data_in(mdata_in),
      .data_in_valid(data_for_max_valid),
      .data_in_ready(data_for_max_ready),
      .data_out(log2_max_value),
      .data_out_valid(log2_max_value_valid),
      .data_out_ready(log2_max_value_ready)
  );

  // =============================
  // FIFO
  // =============================


  if (FIFO_DEPTH == 0) begin

    always_comb begin
      mbuffer_data_for_out = mdata_in;
      ebuffer_data_for_out = edata_in;
      buffer_data_for_out_valid = data_for_out_valid;
      data_for_out_ready = buffer_data_for_out_ready;
    end

  end else begin

    unpacked_mx_fifo #(
        .DEPTH(FIFO_DEPTH),
        .MAN_WIDTH(IN_MAN_WIDTH),
        .EXP_WIDTH(IN_EXP_WIDTH),
        .IN_SIZE(BLOCK_SIZE)
    ) ff_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mdata_in),
        .edata_in(edata_in),
        .data_in_valid(data_for_out_valid),
        .data_in_ready(data_for_out_ready),
        .mdata_out(mbuffer_data_for_out),
        .edata_out(ebuffer_data_for_out),
        .data_out_valid(buffer_data_for_out_valid),
        .data_out_ready(buffer_data_for_out_ready)
    );

  end

  // =============================
  // Handshake Signals
  // =============================

  join2 join_inst (
      .data_in_ready ({buffer_data_for_out_ready, log2_max_value_ready}),
      .data_in_valid ({buffer_data_for_out_valid, log2_max_value_valid}),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );

  // =============================
  // Compute Output Exponent
  // =============================

  assign edata_out_full = log2_max_value - IN_MAN_WIDTH + 2 + ebuffer_data_for_out - EBIAS_IN + EBIAS_OUT;

  always_comb begin

    if (edata_out_full >= (1 << OUT_EXP_WIDTH)) edata_out = (1 << OUT_EXP_WIDTH) - 1;

    else if (edata_out_full < 0) edata_out = 0;

    else edata_out = edata_out_full;

  end

  // =============================
  // Compute Shift Value
  // =============================

  localparam SHIFT_WIDTH = max($clog2(IN_MAN_WIDTH), $clog2(OUT_MAN_WIDTH), 0) + 1;
  logic signed [SHIFT_WIDTH - 1:0] shift_value;
  assign shift_value = edata_out_full - edata_out - log2_max_value + OUT_MAN_WIDTH - 2;


  // =============================
  // Compute Output Mantissa
  // =============================

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin

    always_comb begin

      if (shift_value >= 0 && data_out_valid) begin

        if (mdata_in[i] >= (1 << (OUT_MAN_WIDTH - shift_value)))
          mdata_out[i] = {1'b0, {(OUT_MAN_WIDTH - 2) {1'b1}}};

        else if (-mdata_in[i] >= (1 << (OUT_MAN_WIDTH - shift_value)))
          mdata_out[i] = {1'b1, {(OUT_MAN_WIDTH - 3) {1'b0}}, 1'b1};

        else mdata_out[i] = mbuffer_data_for_out[i] <<< shift_value;

      end else mdata_out[i] = mbuffer_data_for_out[i] >>> -shift_value;

    end

  end

endmodule

// =============================
// Max function
// =============================

function [31:0] max;
  input [31:0] x, y, z;
  begin
    max = (x > y && x > z) ? x : (y > z) ? y : z;
  end
endfunction
