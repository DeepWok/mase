`timescale 1ns / 1ps

module skid_buffer #(
    parameter DATA_WIDTH = 32
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH - 1:0] data_in,
    input  logic                    data_in_valid,
    output logic                    data_in_ready,

    output logic [DATA_WIDTH - 1:0] data_out,
    output logic                    data_out_valid,
    input  logic                    data_out_ready
);
  // feed the data_out either from
  // data_in or a buffered copy of data_in

  logic [DATA_WIDTH - 1:0] data_buffer_out;
  logic data_buffer_wren;
  always_ff @(posedge clk)
    if (rst) data_buffer_out <= 0;
    else if (data_buffer_wren) data_buffer_out <= data_in;

  logic data_out_wren;
  logic use_buffered_data;
  logic [DATA_WIDTH - 1:0] selected_data;
  assign selected_data = (use_buffered_data) ? data_buffer_out : data_in;
  always_ff @(posedge clk)
    if (rst) data_out <= 0;
    else if (data_out_wren) data_out <= selected_data;

  // control path
  // skid buffer has 4 states
  // 1. Empty
  // 2. Busy, holding data in the main register, but not transferring to output
  // 3. Full, both two registers were hold

  enum {
    EMPTY,
    BUSY,
    FULL
  }
      state, state_next;

  always_ff @(posedge clk) begin : handshake
    if (rst) begin
      data_in_ready  <= 0;
      data_out_valid <= 0;
    end else begin
      /* verilator lint_off WIDTH */
      data_in_ready  <= (state_next != FULL);
      data_out_valid <= (state_next != EMPTY);
      /* verilator lint_on WIDTH */
    end
  end
  logic insert, remove;
  always_comb begin
    insert = (data_in_valid && data_in_ready);
    remove = (data_out_valid && data_out_ready);
  end

  logic load, flow, fill, flush, unload;
  always_comb begin
    load   = (state == EMPTY) && ({insert, remove} == 2'b10);
    flow   = (state == BUSY) && ({insert, remove} == 2'b11);
    fill   = (state == BUSY) && ({insert, remove} == 2'b10);
    unload = (state == BUSY) && ({insert, remove} == 2'b01);
    flush  = (state == FULL) && ({insert, remove} == 2'b01);
  end

  always_comb
    /* verilator lint_off WIDTH */
    case (state)
      EMPTY: state_next = (load) ? BUSY : state;
      BUSY: state_next = (fill) ? FULL : (flow) ? BUSY : (unload) ? EMPTY : state;
      FULL: state_next = (flush) ? BUSY : state;
      default: state_next = state;
    endcase
  always_ff @(posedge clk)
    if (rst) state <= EMPTY;
    else state <= state_next;
  /* verilator lint_on WIDTH */
  always_comb begin
    data_out_wren     = (load == 1'b1) || (flow == 1'b1) || (flush == 1'b1);
    data_buffer_wren  = (fill == 1'b1);
    use_buffered_data = (flush == 1'b1);
  end
endmodule
