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
  logic [IN_WIDTH-1:0] buffer;
  // The shift register stores the validity of the data in the buffer 
  logic shift_reg;

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
    if (rst) buffer <= 0;
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
/* verilator lint_off DECLFILENAME */
module unpacked_register_slice #(
    parameter IN_WIDTH = 32,
    parameter IN_SIZE = 16,
    parameter type MYDATA = logic [IN_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA data_in_data [IN_SIZE-1:0],
    input  logic  data_in_valid,
    output logic  data_in_ready,

    output MYDATA data_out_data [IN_SIZE-1:0],
    output logic  data_out_valid,
    input  logic  data_out_ready
);
  logic [IN_WIDTH * IN_SIZE - 1 : 0] data_in_flatten;
  logic [IN_WIDTH * IN_SIZE - 1 : 0] data_out_flatten;
  for (genvar i = 0; i < IN_SIZE; i++) begin
    assign data_in_flatten[i*IN_WIDTH+IN_WIDTH-1:i*IN_WIDTH] = data_in_data[i];
    assign data_out_data[i] = data_out_flatten[i*IN_WIDTH+IN_WIDTH-1:i*IN_WIDTH];
  end
  register_slice #(
      .IN_WIDTH(IN_WIDTH * IN_SIZE)
  ) register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_in_data  (data_in_flatten),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready),
      .data_out_data (data_out_flatten)
  );
endmodule
