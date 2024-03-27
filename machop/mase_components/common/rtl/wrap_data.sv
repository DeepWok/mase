module wrap_data #(
    parameter IN_WIDTH = 32,
    parameter WRAP_Y = 1,
    parameter IN_Y = 2,
    parameter IN_X = 10,
    parameter UNROLL_IN_X = 5
) (
    input clk,
    rst,
    input logic [IN_WIDTH - 1:0] data_in[UNROLL_IN_X -1:0],
    input logic data_in_valid,
    output logic data_in_ready,
    input logic [IN_WIDTH - 1:0] wrap_in[UNROLL_IN_X -1:0],
    input logic wrap_in_valid,
    output logic wrap_in_ready,
    output logic [IN_WIDTH - 1:0] data_out[UNROLL_IN_X -1:0],
    output logic data_out_valid,
    input logic data_out_ready
);

  parameter ITER_X = IN_X / UNROLL_IN_X;
  parameter ITER_Y = IN_Y + WRAP_Y;
  parameter Y_WIDTH = $clog2(ITER_Y);
  parameter X_WIDTH = $clog2(ITER_X);

  enum {
    WRAP,
    DATA
  } mode;
  /* verilator lint_off LITENDIAN */
  logic [Y_WIDTH-1:0] in_y, in_y_next;
  logic [X_WIDTH-1:0] in_x, in_x_next;
  /* verilator lint_on LITENDIAN */
  // data position arrange
  /* verilator lint_off WIDTH */
  always_comb begin
    // consider the input matrix
    // row input first, 
    // only if input the whole row*channel,
    // then input the next row, so column + 1
    if (in_y == ITER_Y - 1 && in_x == ITER_X - 1) begin
      in_x_next = 0;
      in_y_next = 0;
    end else if (in_x == ITER_X - 1) begin
      in_x_next = 0;
      in_y_next = in_y + 1;
    end else begin
      in_x_next = in_x + 1;
      in_y_next = in_y;
    end
  end
  // always_ff @(posedge clk)
  always_ff @(posedge clk)
    if (rst) begin
      in_x <= 0;
      in_y <= 0;
      mode <= WRAP;
    end else begin
      if (data_out_valid && data_out_ready) begin
        in_x <= in_x_next;
        in_y <= in_y_next;
        if (in_x_next == 0 && in_y_next == 0) mode <= WRAP;
        else if (in_y_next == WRAP_Y) mode <= DATA;
      end
    end
  /* verilator lint_on WIDTH */
  assign data_out_valid = (mode == DATA) ? data_in_valid : wrap_in_valid;
  // we can take input if our buffer is not full, or if output is ready.
  assign data_in_ready = mode == DATA && data_out_ready;
  assign wrap_in_ready = mode == WRAP && data_out_ready;
  assign data_out = (mode == DATA) ? data_in : wrap_in;

endmodule
