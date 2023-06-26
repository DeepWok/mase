`timescale 1ns / 1ps
module register_slice #(
    parameter IN_WIDTH = 32,
    parameter type MYDATA = logic [IN_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA data_in_data,
    input  logic  data_in_valid,
    output logic  data_in_ready,

    output MYDATA data_out_data,
    output logic  data_out_valid,
    input  logic  data_out_ready
);

  // The buffer stores the intermeidate data being computed in the register slice
  MYDATA buffer;
  // The shift register stores the validity of the data in the buffer 
  logic  shift_reg;

  // shift_register 
  always_ff @(posedge clk) begin
    if (rst) shift_reg <= 1'b0;
    else begin
      // no backpressure or buffer empty
      if (data_out_ready || !shift_reg) shift_reg <= data_in_valid;
      else shift_reg <= shift_reg;
    end
  end

  // buffer 
  always_ff @(posedge clk) begin
    // backpressure && valid output
    if (!data_out_ready && data_out_valid) buffer <= buffer;
    else buffer <= data_in_data;
  end

  always_comb begin
    // empty buffer or no back pressure
    data_in_ready  = (~shift_reg) | data_out_ready;
    // dummy data_iniring 
    data_out_valid = shift_reg;
    data_out_data  = buffer;
  end

endmodule
