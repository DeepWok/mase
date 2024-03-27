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
/* verilator lint_off DECLFILENAME */
module unpacked_register_slice #(
    parameter DATA_WIDTH = 32,
    parameter IN_SIZE = 16,
    parameter type MYDATA = logic [DATA_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA in_data [IN_SIZE-1:0],
    input  logic  in_valid,
    output logic  in_ready,

    output MYDATA out_data [IN_SIZE-1:0],
    output logic  out_valid,
    input  logic  out_ready
);
  logic [DATA_WIDTH * IN_SIZE - 1 : 0] data_in_flatten;
  logic [DATA_WIDTH * IN_SIZE - 1 : 0] data_out_flatten;
  for (genvar i = 0; i < IN_SIZE; i++) begin
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = in_data[i];
    assign out_data[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end
  register_slice #(
      .DATA_WIDTH(DATA_WIDTH * IN_SIZE)
  ) register_slice (
      .clk      (clk),
      .rst      (rst),
      .in_valid (in_valid),
      .in_ready (in_ready),
      .in_data  (data_in_flatten),
      .out_valid(out_valid),
      .out_ready(out_ready),
      .out_data (data_out_flatten)
  );
endmodule
