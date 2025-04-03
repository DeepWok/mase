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
    input  logic [IN_MAN_WIDTH-1:0] mdata_in     [BLOCK_SIZE-1:0],
    input  logic [IN_EXP_WIDTH-1:0] edata_in,
    input  logic                    data_in_valid,
    output logic                    data_in_ready,

    // Output Data
    output logic [OUT_MAN_WIDTH-1:0] mdata_out     [BLOCK_SIZE-1:0],
    output logic [OUT_EXP_WIDTH-1:0] edata_out,
    output logic                     data_out_valid,
    input  logic                     data_out_ready
);

  // =============================
  // Check Parameters
  // =============================

  initial begin
    assert (IN_EXP_WIDTH > 2)
    else $fatal("IN_EXP_WIDTH must be greater than 2");
    assert (IN_MAN_WIDTH > 3)
    else $fatal("IN_MAN_WIDTH must be greater than 3");
    assert (OUT_EXP_WIDTH > 2)
    else $fatal("OUT_EXP_WIDTH must be greater than 2");
    assert (OUT_MAN_WIDTH > 3)
    else $fatal("OUT_MAN_WIDTH must be greater than 3");
  end

  // =============================
  // Internal Signals
  // =============================

  logic data_for_max_valid, data_for_max_ready, data_for_out_valid, data_for_out_ready;
  logic signed [IN_MAN_WIDTH-1:0] mbuffer_data_for_out[BLOCK_SIZE-1:0];
  logic [IN_MAN_WIDTH-1:0] fifo_out[BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] ebuffer_data_for_out;
  logic buffer_data_for_out_valid, buffer_data_for_out_ready;

  localparam LOG2_WIDTH = $clog2(IN_MAN_WIDTH) + 1;
  logic [LOG2_WIDTH - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  localparam EBIAS_OUT = 2 ** (OUT_EXP_WIDTH - 1) - 1;
  localparam EBIAS_IN = 2 ** (IN_EXP_WIDTH - 1) - 1;
  localparam LOSSLESSS_EDATA_WIDTH = max(LOG2_WIDTH, IN_EXP_WIDTH, OUT_EXP_WIDTH) + 2;
  localparam FIFO_DEPTH = $clog2(BLOCK_SIZE);
  logic signed [LOSSLESSS_EDATA_WIDTH - 1:0] edata_out_full;

  localparam MAX_DATA_OUT = 2 ** (OUT_MAN_WIDTH - 1) - 1;
  localparam MIN_DATA_OUT = -MAX_DATA_OUT;

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
      for (int i = 0; i < BLOCK_SIZE; i++) begin
        mbuffer_data_for_out[i] = $signed(mdata_in[i]);
      end
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
        .mdata_out(fifo_out),
        .edata_out(ebuffer_data_for_out),
        .data_out_valid(buffer_data_for_out_valid),
        .data_out_ready(buffer_data_for_out_ready)
    );

    always_comb begin
      for (int i = 0; i < BLOCK_SIZE; i++) begin
        mbuffer_data_for_out[i] = $signed(fifo_out[i]);
      end
    end

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

    if (log2_max_value == 0) edata_out = 0;

    else if (edata_out_full >= (1 << OUT_EXP_WIDTH)) edata_out = (1 << OUT_EXP_WIDTH) - 1;

    else if (edata_out_full < 0) edata_out = 0;

    else edata_out = edata_out_full;

  end

  // =============================
  // Compute Shift Value
  // =============================

  localparam SHIFT_WIDTH = max(LOSSLESSS_EDATA_WIDTH, OUT_MAN_WIDTH, 0);
  logic signed [SHIFT_WIDTH - 1:0] shift_value;
  logic [IN_MAN_WIDTH - 1:0] max_value;

  always_comb begin

    shift_value = $signed(edata_out_full - edata_out) + $signed(OUT_MAN_WIDTH - log2_max_value - 2);

    max_value = (1 << (OUT_MAN_WIDTH - shift_value - 1));

    if (max_value == 0) max_value = 2 ** (IN_MAN_WIDTH - 1);

  end

  // =============================
  // Compute Output Mantissa
  // =============================

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin

    always_comb begin

      if (mbuffer_data_for_out[i] == 0) mdata_out[i] = 0;

      else if ((shift_value > 0) && (shift_value >= OUT_MAN_WIDTH))
        if (mbuffer_data_for_out[i] < 0) mdata_out[i] = MIN_DATA_OUT;
        else mdata_out[i] = MAX_DATA_OUT;

      // This is really stupid, but system verilog has poor support for signed arithmetic of large numbers
      // So -shift_value != twos_complement(shift_value) hence:
      else if ((shift_value < 0) && (twos_complement(shift_value) >= IN_MAN_WIDTH))
        if (mbuffer_data_for_out[i] < 0) mdata_out[i] = -1;
        else mdata_out[i] = 0;

      else if ((mbuffer_data_for_out[i] > 0) && (mbuffer_data_for_out[i] >= max_value))
        mdata_out[i] = MAX_DATA_OUT;

      else if ((mbuffer_data_for_out[i] < 0) && (twos_complement(
              mbuffer_data_for_out[i]
          ) >= max_value))
        mdata_out[i] = MIN_DATA_OUT;

      else if (shift_value >= 0) mdata_out[i] = mbuffer_data_for_out[i] <<< shift_value;

      else mdata_out[i] = mbuffer_data_for_out[i] >>> twos_complement(shift_value);

    end

  end

  // =============================
  // two's complement
  // =============================

  localparam TWOS_COMPLEMENT_WIDTH = max(IN_MAN_WIDTH, SHIFT_WIDTH, 0);

  function [TWOS_COMPLEMENT_WIDTH - 1:0] twos_complement;
    input [TWOS_COMPLEMENT_WIDTH - 1:0] x;
    begin
      twos_complement = ~x + 1;
    end
  endfunction

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

