`timescale 1ns / 1ps
module register_slice #(
    parameter DATA_WIDTH = 32,
    parameter type MYDATA = logic [DATA_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA in_data,
    input  logic  in_valid,
    output logic  in_ready,

    output MYDATA out_data,
    output logic  out_valid,
    input  logic  out_ready
);

  // The buffer stores the intermeidate data being computed in the register slice
  logic [DATA_WIDTH-1:0] buffer;
  // The shift register stores the validity of the data in the buffer
  logic shift_reg;

  // shift_register
  always_ff @(posedge clk) begin
    if (rst) shift_reg <= 1'b0;
    else begin
      // no backpressure or buffer empty
      if (out_ready || !shift_reg) shift_reg <= in_valid;
      else shift_reg <= shift_reg;
    end
  end

  // buffer
  always_ff @(posedge clk) begin
    if (rst) buffer <= 0;
    // backpressure && valid output
    if (!out_ready && out_valid) buffer <= buffer;
    else buffer <= in_data;
  end

  always_comb begin
    // empty buffer or no back pressure
    in_ready  = (~shift_reg) | out_ready;
    // dummy data_iniring
    out_valid = shift_reg;
    out_data  = buffer;
  end

endmodule
