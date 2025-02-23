`timescale 1ns / 1ps

module sequential_max #(
    parameter IN_DEPTH  = 4,
    parameter IN_WIDTH  = 32
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] data_in,
    input  logic                data_in_valid,
    output logic                data_in_ready,

    output logic [IN_WIDTH-1:0] data_out,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);
  logic [IN_WIDTH-1:0] reg_in;
  logic reg_in_valid, reg_in_ready;

  skid_buffer #(
      .DATA_WIDTH(IN_WIDTH)
  ) register_slice (
      .data_in(reg_in),
      .data_in_valid(reg_in_valid),
      .data_in_ready(reg_in_ready),
      .*
  );
  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(IN_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  /* verilator lint_off WIDTH */
  assign data_in_ready = (counter != IN_DEPTH) || reg_in_ready;
  assign reg_in_valid  = (counter == IN_DEPTH);
  /* verilator lint_on WIDTH */

  // counter logic
  always_ff @(posedge clk) begin
    if (rst) begin
      counter <= '0;
    end else begin
      if (reg_in_valid && reg_in_ready) begin
        // Reset counter or start new sequence
        counter <= data_in_valid ? 1'b1 : '0;
      end else if (data_in_valid && data_in_ready) begin
        // Continue counting inputs
        counter <= counter + 1'b1;
      end
    end
  end

  // max value tracking logic
  always_ff @(posedge clk) begin
    if (rst) begin
      reg_in <= '0;
    end else begin
      if (reg_in_valid && reg_in_ready) begin
        // Reset or start new maximum tracking
        reg_in <= data_in_valid ? data_in : '0;
      end else if (data_in_valid && data_in_ready) begin
        // Update maximum if new value is larger
        reg_in <= ($signed(data_in) > $signed(reg_in)) ? data_in : reg_in;
      end
    end
  end

endmodule
