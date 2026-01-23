`timescale 1ns / 1ps
/*
Module      : Mxint cast
Description : MxInt Cast between Layers.
*/
module mxint_cast #(
    parameter IN_MAN_WIDTH = 1,
    parameter IN_MAN_FRAC_WIDTH = IN_MAN_WIDTH - 1,
    parameter IN_EXP_WIDTH = 1,
    parameter OUT_MAN_WIDTH = 1,
    parameter OUT_EXP_WIDTH = 1,
    parameter BLOCK_SIZE = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                     clk,
    input  logic                     rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_MAN_WIDTH-1:0] mdata_in      [BLOCK_SIZE-1:0],
    input  logic [ IN_EXP_WIDTH-1:0] edata_in,
    input  logic                     data_in_valid,
    output logic                     data_in_ready,
    output logic [OUT_MAN_WIDTH-1:0] mdata_out     [BLOCK_SIZE-1:0],
    output logic [OUT_EXP_WIDTH-1:0] edata_out,
    output logic                     data_out_valid,
    input  logic                     data_out_ready
);
  //get max_abs_value of input
  logic data_for_max_valid, data_for_max_ready, data_for_out_valid, data_for_out_ready;
  split2 #() split_i (
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out_valid({data_for_max_valid, data_for_out_valid}),
      .data_out_ready({data_for_max_ready, data_for_out_ready})
  );
  logic [IN_MAN_WIDTH-1:0] mbuffer_data_for_out [BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] ebuffer_data_for_out;
  logic buffer_data_for_out_valid, buffer_data_for_out_ready;

  localparam LOG2_WIDTH = $clog2(IN_MAN_WIDTH) + 1;
  logic [LOG2_WIDTH - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  localparam LOSSLESSS_EDATA_WIDTH = 
    (LOG2_WIDTH > IN_EXP_WIDTH && LOG2_WIDTH > OUT_EXP_WIDTH) ? LOG2_WIDTH + 2 :
    (IN_EXP_WIDTH > OUT_EXP_WIDTH) ? IN_EXP_WIDTH + 2:
    OUT_EXP_WIDTH + 2;

  localparam FIFO_DEPTH = $clog2(BLOCK_SIZE);
  logic [LOSSLESSS_EDATA_WIDTH - 1:0] edata_out_full;
  localparam SHIFT_WIDTH = (OUT_EXP_WIDTH > IN_EXP_WIDTH) ? OUT_EXP_WIDTH + 1 : IN_EXP_WIDTH + 1;
  logic [SHIFT_WIDTH - 1:0] shift_value;
  logic [SHIFT_WIDTH - 1:0] abs_shift_value;
  // we dont need to implement full shift here, because we'll clamp in the final.
  // in order to avoid shift loss, we set the shift_data_width = OUT_MAN_WIDTH + 1.
  localparam SHIFT_DATA_WIDTH = OUT_MAN_WIDTH + 1;

  logic [SHIFT_DATA_WIDTH - 1:0] shift_buffer_data_for_out[BLOCK_SIZE - 1:0];
  logic [SHIFT_DATA_WIDTH - 1:0] shift_data[BLOCK_SIZE - 1:0][SHIFT_DATA_WIDTH - 1:0];
  logic [$clog2(SHIFT_DATA_WIDTH) - 1:0] real_shift_value;
  log2_max_abs #(
      .IN_SIZE (BLOCK_SIZE),
      .IN_WIDTH(IN_MAN_WIDTH)
  ) max_bas_i (
      .clk,
      .rst,
      .data_in_0(mdata_in),
      .data_in_0_valid(data_for_max_valid),
      .data_in_0_ready(data_for_max_ready),
      .data_out_0(log2_max_value),
      .data_out_0_valid(log2_max_value_valid),
      .data_out_0_ready(log2_max_value_ready)
  );

  if (FIFO_DEPTH == 0) begin : register
    mxint_register_slice #(
        .DATA_PRECISION_0($bits(mbuffer_data_for_out[0])),
        .DATA_PRECISION_1($bits(ebuffer_data_for_out)),
        .IN_NUM(BLOCK_SIZE)
    ) register_slice (
        .clk           (clk),
        .rst           (rst),
        .mdata_in      (mdata_in),
        .edata_in      (edata_in),
        .data_in_valid (data_for_out_valid),
        .data_in_ready (data_for_out_ready),
        .mdata_out     (mbuffer_data_for_out),
        .edata_out     (ebuffer_data_for_out),
        .data_out_valid(buffer_data_for_out_valid),
        .data_out_ready(buffer_data_for_out_ready)
    );
  end else begin : data_buffer
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
  join2 #() join_inst (
      .data_in_ready ({buffer_data_for_out_ready, log2_max_value_ready}),
      .data_in_valid ({buffer_data_for_out_valid, log2_max_value_valid}),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
  assign edata_out_full = $signed(
      log2_max_value
  ) + $signed(
      ebuffer_data_for_out
  ) - IN_MAN_FRAC_WIDTH;
  // clamp 
  signed_clamp #(
      .IN_WIDTH (LOSSLESSS_EDATA_WIDTH),
      .OUT_WIDTH(OUT_EXP_WIDTH)
  ) exp_clamp (
      .in_data (edata_out_full),
      .out_data(edata_out)
  );
  optimized_variable_shift #(
      .IN_WIDTH(IN_MAN_WIDTH),
      .SHIFT_WIDTH(SHIFT_WIDTH),
      .OUT_WIDTH(OUT_MAN_WIDTH),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) ovshift_inst (
      .data_in(mbuffer_data_for_out),
      .shift_value(shift_value),
      .data_out(mdata_out)
  );
  assign shift_value = $signed(
      edata_out
  ) - $signed(
      ebuffer_data_for_out
  ) + IN_MAN_FRAC_WIDTH - (OUT_MAN_WIDTH - 1);
  //   assign abs_shift_value = (shift_value[SHIFT_WIDTH-1]) ? (~shift_value + 1) : shift_value;
  //   assign real_shift_value = (abs_shift_value < SHIFT_DATA_WIDTH)? abs_shift_value: SHIFT_DATA_WIDTH - 1;

  //   for (genvar i = 0; i < BLOCK_SIZE; i++) begin
  //     for (genvar j = 0; j < SHIFT_DATA_WIDTH; j++) begin
  //       always_comb begin
  //         shift_data[i][j] = (shift_value[SHIFT_WIDTH-1]) ? $signed(
  //             mbuffer_data_for_out[i]
  //         ) <<< j : $signed(
  //             mbuffer_data_for_out[i]
  //         ) >>> j;
  //       end
  //     end
  //     assign shift_buffer_data_for_out[i] = shift_data[i][real_shift_value];
  //   end
  //   for (genvar i = 0; i < BLOCK_SIZE; i++) begin
  //     signed_clamp #(
  //         .IN_WIDTH (OUT_MAN_WIDTH + 1),
  //         .OUT_WIDTH(OUT_MAN_WIDTH)
  //     ) exp_clamp (
  //         .in_data (shift_buffer_data_for_out[i]),
  //         .out_data(mdata_out[i])
  //     );
  //   end
endmodule
// function int max(input int x, y, z);
//   begin
//     if (x > y && x > z) max = x;
//     else if (y > z) max = y;
//     else max = z;
//   end
// endfunction
