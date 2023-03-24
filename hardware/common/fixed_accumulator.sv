`timescale 1ns / 1ps
module fixed_accumulator #(
    parameter IN_DEPTH  = 4,
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_DEPTH) + IN_WIDTH
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic [ IN_WIDTH-1:0] data_in,
    input  logic                 data_in_valid,
    output logic                 data_in_ready,
    output logic [OUT_WIDTH-1:0] data_out,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);

  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(IN_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  // Sign extension before feeding into the accumulator
  logic [  OUT_WIDTH-1:0] data_in_sext;
  assign data_in_sext   = {{(OUT_WIDTH - IN_WIDTH) {data_in[IN_WIDTH-1]}}, data_in};

  /* verilator lint_off WIDTH */
  assign data_in_ready  = (counter != IN_DEPTH) || data_out_ready;
  assign data_out_valid = (counter == IN_DEPTH);
  /* verilator lint_on WIDTH */

  // counter
  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (data_out_valid) begin
        if (data_out_ready) begin
          if (data_in_valid) counter <= 1;
          else counter <= 0;
        end
      end else if (data_in_valid && data_in_ready) counter <= counter + 1;
    end

  // data_out 
  always_ff @(posedge clk)
    if (rst) data_out <= '0;
    else begin
      if (data_out_valid) begin
        if (data_out_ready) begin
          if (data_in_valid) data_out <= data_in_sext;
          else data_out <= '0;
        end
      end else if (data_in_valid && data_in_ready) data_out <= data_out + data_in_sext;
    end

endmodule
