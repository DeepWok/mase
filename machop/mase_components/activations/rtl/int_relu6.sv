`timescale 1ns / 1ps
module int_relu6 #(
    parameter NUM = -1,
    parameter ACT_WIDTH = -1,
    parameter ACT_BIAS = -1
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic [NUM-1:0][ACT_WIDTH-1:0] in,
    input  logic                          in_valid,
    output logic                          in_ready,
    output logic [NUM-1:0][ACT_WIDTH-1:0] out,
    output logic                          out_valid,
    input  logic                          out_ready
);

  localparam CAP = 6 << ACT_BIAS;

  logic [NUM-1:0][ACT_WIDTH-1:0] relued;

  for (genvar i = 0; i < NUM; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if (in[i][ACT_WIDTH-1]) relued[i] = '0;
      // cap to 6            else
      if (in[i] >= CAP) relued[i] = CAP;
      else relued[i] = in[i];
    end
  end

  register_slice #(
      .DATA_WIDTH ($bits(relued)),
      .REVERSE   (1'b0)
  ) register_slice (
      .clk    (clk),
      .rst    (rst),
      .w_valid(in_valid),
      .w_ready(in_ready),
      .w_data (relued),
      .r_valid(out_valid),
      .r_ready(out_ready),
      .r_data (out)
  );

endmodule
