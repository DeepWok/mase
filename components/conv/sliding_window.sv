`timescale 1ns / 1ps
/* verilator lint_off DECLFILENAME */
module sliding_window_buffer #(
    parameter DATA_WIDTH = 32,

    parameter IMG_WIDTH  = 4,
    parameter IMG_HEIGHT = 3,

    parameter KERNEL_WIDTH  = 3,
    parameter KERNEL_HEIGHT = 2,

    parameter CHANNELS = 2,

    // X_WIDTH means which line here is
    parameter X_WIDTH = $clog2(IMG_WIDTH) + 1,
    parameter Y_WIDTH = $clog2(IMG_HEIGHT) + 1,
    parameter C_WIDTH = $clog2(CHANNELS) + 1
) (
    input logic clk,
    input logic rst,

    // Input shape is IMG_HEIGHT * IMG_WIDTH * CHANNELS, not unrolled.
    // Here data_in means input DATA_WIDTH length input one time, and need to organized it to KERNEL_HEIGHT * KERNEL_WIDTH
    input logic [DATA_WIDTH - 1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,

    // Output shape is IMG_HEIGH * IMG_WIDTH * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH,
    // CHANNELS are not unrolled and KERNEL_HEIGHT and KERNEL_WIDTH are unrolled

    output logic [DATA_WIDTH - 1:0] data_out[KERNEL_HEIGHT * KERNEL_WIDTH- 1:0],
    output logic [X_WIDTH-1:0] out_x,
    output logic [Y_WIDTH-1:0] out_y,
    output logic [C_WIDTH-1:0] out_c,
    output logic data_out_valid,
    input logic data_out_ready
);

  localparam LINE_WIDTH = IMG_WIDTH * CHANNELS;
  localparam CX_WIDTH = $clog2(LINE_WIDTH);

  localparam BUF_SIZE = LINE_WIDTH * (KERNEL_HEIGHT - 1) + (KERNEL_WIDTH - 1) * CHANNELS + 1;

  enum {
    FILL,
    PIPELINE,
    DRAIN
  } mode;

  // Current output pixel locates at BUF_SIZE/2.
  logic [DATA_WIDTH-1:0] shift_reg[BUF_SIZE-1:0];

  logic [CX_WIDTH-1:0] in_cx, in_cx_next;
  logic [X_WIDTH-1:0] out_x_next;
  logic [Y_WIDTH-1:0] in_y, in_y_next, out_y_next;
  logic [C_WIDTH-1:0] out_c_next;
  // data position arrange
  /* verilator lint_off WIDTH */
  always_comb begin
    // consider the input matrix
    // row input first, 
    // only if input the whole row*channel,
    // then input the next row, so column + 1
    if (in_cx == LINE_WIDTH - 1) begin
      in_cx_next = 0;
      in_y_next  = in_y + 1;
    end else begin
      in_cx_next = in_cx + 1;
      in_y_next  = in_y;
    end

    // consider the output matrix
    // but don't know why out c vary like this ???????????????????
    if (out_c == CHANNELS - 1) begin
      out_c_next = 0;
      if (out_x == IMG_WIDTH - 1) begin
        out_x_next = 0;
        out_y_next = out_y + 1;
      end else begin
        out_x_next = out_x + 1;
        out_y_next = out_y;
      end
    end else begin
      out_c_next = out_c + 1;
      out_x_next = out_x;
      out_y_next = out_y;
    end
  end
  logic [DATA_WIDTH-1:0] next_shift[BUF_SIZE-1:0];
  logic [DATA_WIDTH-1:0] shift_initialize[BUF_SIZE-1:0];

  assign next_shift[BUF_SIZE-1] = data_in;
  for (genvar k = 0; k < BUF_SIZE - 1; k++) assign next_shift[k] = shift_reg[k+1];

  for (genvar k = 0; k < BUF_SIZE; k++) assign shift_initialize[k] = 0;
  always_ff @(posedge clk)
    if (rst) shift_reg[BUF_SIZE-1:0] <= shift_initialize;
    else if ((data_in_valid && data_in_ready) || (data_in_valid && data_in_ready))
      shift_reg[BUF_SIZE-1:0] <= next_shift;

  // always_ff @(posedge clk)
  always_ff @(posedge clk)
    if (rst) begin
      in_cx <= 0;
      in_y  <= 0;
      out_x <= 0;
      out_y <= 0;
      out_c <= 0;
      mode  <= FILL;
    end else begin
      if (data_in_valid && data_in_ready) begin
        in_cx <= in_cx_next;
        in_y  <= in_y_next;
        // Move from FILL mode to PIPELINE mode once we can start outputting
        if (in_y == KERNEL_HEIGHT - 1 && in_cx == (KERNEL_WIDTH - 1) * CHANNELS) mode <= PIPELINE;

        // When we have input all the data
        // Move from PIPELINE mode to DRAIN mode
        if (in_y == IMG_HEIGHT - 1 && in_cx == LINE_WIDTH - 1) begin
          mode  <= DRAIN;
          in_cx <= 0;
          in_y  <= 0;
        end
      end

      if (data_out_valid && data_out_ready) begin
        out_c <= out_c_next;
        out_x <= out_x_next;
        out_y <= out_y_next;

        //When output full, initialize
        if (out_y == IMG_HEIGHT - KERNEL_HEIGHT && out_x == IMG_WIDTH - KERNEL_WIDTH && out_c == CHANNELS - 1)begin
          mode  <= FILL;
          out_x <= 0;
          out_y <= 0;
          out_c <= 0;
        end
      end
    end

  assign data_out_valid = mode == DRAIN || (mode == PIPELINE && data_in_valid);
  // we can take input if our buffer is not full, or if output is ready.
  assign data_in_ready  = mode == FILL || (mode == PIPELINE && data_out_ready);

  for (genvar j = 0; j < KERNEL_HEIGHT; j++) begin : line
    for (genvar i = 0; i < KERNEL_WIDTH; i++) begin : word
      always_comb begin
        data_out[j*KERNEL_WIDTH+i] = shift_reg[j*LINE_WIDTH+i*CHANNELS];
        $display(j * LINE_WIDTH + i * CHANNELS);
      end
    end
  end


endmodule

module sliding_window_stride #(
    parameter DATA_WIDTH = 32,

    parameter IMG_WIDTH  = 4,
    parameter IMG_HEIGHT = 3,

    parameter KERNEL_WIDTH  = 3,
    parameter KERNEL_HEIGHT = 2,

    parameter CHANNELS = 2,
    parameter STRIDE   = 2
) (
    input logic clk,
    input logic rst,

    // Input shape is IMG_HEIGHT * IMG_WIDTH * CHANNELS, not unrolled.
    // Here data_in means input DATA_WIDTH length input one time, and need to organized it to KERNEL_HEIGHT * KERNEL_WIDTH
    input logic [DATA_WIDTH - 1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,

    // Output shape is IMG_HEIGH * IMG_WIDTH * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH,
    // CHANNELS are not unrolled and KERNEL_HEIGHT and KERNEL_WIDTH are unrolled

    output logic [DATA_WIDTH - 1:0] data_out[KERNEL_HEIGHT * KERNEL_WIDTH- 1:0],
    output logic data_out_valid,
    input logic data_out_ready
);
  localparam X_WIDTH = $clog2(IMG_WIDTH) + 1;
  localparam Y_WIDTH = $clog2(IMG_HEIGHT) + 1;
  localparam C_WIDTH = $clog2(CHANNELS) + 1;

  logic [DATA_WIDTH - 1:0] buffer_data[KERNEL_HEIGHT * KERNEL_WIDTH- 1:0];
  logic [X_WIDTH-1:0] buffer_x;
  logic [Y_WIDTH-1:0] buffer_y;
  logic [C_WIDTH-1:0] buffer_c;
  logic buffer_valid;
  logic buffer_ready;

  sliding_window_buffer #(
      .IMG_WIDTH(IMG_WIDTH),
      .IMG_HEIGHT(IMG_HEIGHT),
      .KERNEL_WIDTH(KERNEL_WIDTH),
      .KERNEL_HEIGHT(KERNEL_HEIGHT),
      .DATA_WIDTH(DATA_WIDTH),
      .CHANNELS(CHANNELS)
  ) buffer (
      .data_out      (buffer_data),
      .out_x         (buffer_x),
      .out_y         (buffer_y),
      .out_c         (buffer_c),
      .data_out_valid(buffer_valid),
      .data_out_ready(buffer_ready),
      .*
  );
  // enable stride == 1
  logic in_range;
  logic sliding_valid, sliding_ready;
  assign in_range = (buffer_x + (KERNEL_WIDTH - 1) < IMG_WIDTH)&&(buffer_y + (KERNEL_HEIGHT-1) < IMG_HEIGHT);

  assign sliding_valid = buffer_valid && in_range;
  assign buffer_ready = sliding_ready || (!in_range);

  // // We can start output when our buffer is filled and the input word is available.
  // detect the change of out_x and out_y to determine whether enable

  logic x_change, y_change, x_en, y_en, x_reset, y_reset;
  always_comb begin
    x_change = (buffer_c == CHANNELS - 1) && buffer_valid && buffer_ready;
    y_change = (buffer_x == IMG_WIDTH - 1)&&(buffer_c == CHANNELS - 1)&&buffer_valid&&buffer_ready;
    case ({
      x_change, y_change
    })
      2'b11:
      if (buffer_y == IMG_HEIGHT - KERNEL_HEIGHT && buffer_x == IMG_WIDTH - KERNEL_WIDTH)
        {x_reset, y_reset, x_en, y_en} = 4'b11xx;
      else {x_reset, y_reset, x_en, y_en} = 4'b10x1;
      2'b10:
      if (buffer_y == IMG_HEIGHT - KERNEL_HEIGHT && buffer_x == IMG_WIDTH - KERNEL_WIDTH)
        {x_reset, y_reset, x_en, y_en} = 4'b11xx;
      else if (buffer_x == IMG_WIDTH - 1) {x_reset, y_reset, x_en, y_en} = 4'b1000;
      else {x_reset, y_reset, x_en, y_en} = 4'b0010;
      2'b01: {x_reset, y_reset, x_en, y_en} = 4'b1001;
      2'b00: {x_reset, y_reset, x_en, y_en} = 4'b0000;

    endcase
  end
  // stride enable
  localparam S_WIDTH = $clog2(STRIDE);
  logic [S_WIDTH:0] count_stride_x;
  logic [S_WIDTH:0] count_stride_y;
  logic stride_enable;

  always_ff @(posedge clk) begin
    if (rst) begin
      count_stride_x <= 0;
      count_stride_y <= 0;
    end
    if (x_change) begin
      if ((count_stride_x == STRIDE - 1) || x_reset) count_stride_x <= 0;
      else if (x_en) count_stride_x <= count_stride_x + 1;

      if (y_reset) count_stride_y <= 0;
      else if (y_change)
        if ((count_stride_y == STRIDE - 1)) count_stride_y <= 0;
        else if (y_en) count_stride_y <= count_stride_y + 1;
    end
  end

  assign stride_enable = ((count_stride_x == 0)) && ((count_stride_y == 0));

  // if stride_enable data went in
  assign data_out_valid = sliding_valid && stride_enable;
  assign sliding_ready = data_out_ready || (!stride_enable);
  assign data_out = buffer_data;


endmodule

module sliding_window #(
    parameter DATA_WIDTH = 32,

    parameter IMG_WIDTH  = 4,
    parameter IMG_HEIGHT = 3,

    parameter KERNEL_WIDTH  = 3,
    parameter KERNEL_HEIGHT = 2,

    parameter PADDING_WIDTH  = 3,
    parameter PADDING_HEIGHT = 2,

    parameter CHANNELS = 2,
    parameter STRIDE   = 2
) (
    input logic clk,
    input logic rst,

    // Input shape is IMG_HEIGHT * IMG_WIDTH * CHANNELS, not unrolled.
    // Here data_in means input DATA_WIDTH length input one time, and need to organized it to KERNEL_HEIGHT * KERNEL_WIDTH
    input logic [DATA_WIDTH - 1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,

    // Output shape is IMG_HEIGH * IMG_WIDTH * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH,
    // CHANNELS are not unrolled and KERNEL_HEIGHT and KERNEL_WIDTH are unrolled

    output logic [DATA_WIDTH - 1:0] data_out[KERNEL_HEIGHT * KERNEL_WIDTH- 1:0],
    output logic data_out_valid,
    input logic data_out_ready
);
  logic [DATA_WIDTH - 1:0] padding_in;
  logic padding_in_valid, padding_in_ready;
  padding #(
      .IMG_WIDTH(IMG_WIDTH),
      .IMG_HEIGHT(IMG_HEIGHT),
      .PADDING_WIDTH(PADDING_WIDTH),
      .PADDING_HEIGHT(PADDING_HEIGHT),
      .DATA_WIDTH(DATA_WIDTH),
      .CHANNELS(CHANNELS)
  ) padding_inst (
      .data_out(padding_in),
      .data_out_valid(padding_in_valid),
      .data_out_ready(padding_in_ready),
      .*
  );

  sliding_window_stride #(
      .IMG_WIDTH(IMG_WIDTH + 2 * PADDING_WIDTH),
      .IMG_HEIGHT(IMG_HEIGHT + 2 * PADDING_HEIGHT),
      .KERNEL_WIDTH(KERNEL_WIDTH),
      .KERNEL_HEIGHT(KERNEL_HEIGHT),
      .DATA_WIDTH(DATA_WIDTH),
      .CHANNELS(CHANNELS),
      .STRIDE(STRIDE)
  ) sws_inst (
      .data_in(padding_in),
      .data_in_valid(padding_in_valid),
      .data_in_ready(padding_in_ready),
      .*
  );

endmodule
