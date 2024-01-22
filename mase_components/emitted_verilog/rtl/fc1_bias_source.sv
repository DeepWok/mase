
// =====================================
//     Mase Hardware
//     Parameter: fc1_bias
//     21/01/2024 17:38:45
// =====================================

`timescale 1 ns / 1 ps
module fc1_bias_rom #(
    parameter DWIDTH   = 80,
    parameter MEM_SIZE = 1,
    parameter AWIDTH   = $clog2(MEM_SIZE) + 1
) (
    input clk,
    input logic [AWIDTH-1:0] addr0,
    input ce0,
    output logic [DWIDTH-1:0] q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemh("top/hardware/rtl/fc1_bias_rom.dat", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module fc1_bias #(
    parameter DATA_WIDTH = 32'd80,
    parameter ADDR_RANGE = 32'd1,
    parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
    input reset,
    input clk,
    input logic [ADDR_WIDTH - 1:0] address0,
    input ce0,
    output logic [DATA_WIDTH - 1:0] q0
);

  fc1_bias_rom fc1_bias_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule


`timescale 1ns / 1ps
module fc1_bias_source #(
    parameter BIAS_TENSOR_SIZE_DIM_0 = 32,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PRECISION_0 = 16,

    parameter BIAS_PARALLELISM_DIM_0 = 1,
    parameter BIAS_PARALLELISM_DIM_1 = 1,
    parameter OUT_DEPTH = BIAS_TENSOR_SIZE_DIM_0 / BIAS_PARALLELISM_DIM_0
) (
    input clk,
    input rst,

    output logic [BIAS_PRECISION_0-1:0] data_out      [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1-1:0],
    output data_out_valid,
    input data_out_ready
);
  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(OUT_DEPTH);
  logic [COUNTER_WIDTH:0] counter;

  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (data_out_ready) begin
        if (counter == OUT_DEPTH - 1) counter <= 0;
        else counter <= counter + 1;
      end
    end

  logic ce0;
  assign ce0 = 1;

  logic [BIAS_PRECISION_0*BIAS_TENSOR_SIZE_DIM_0-1:0] data_vector;
  fc1_bias #(
      .DATA_WIDTH(BIAS_PRECISION_0 * BIAS_TENSOR_SIZE_DIM_0),
      .ADDR_RANGE(OUT_DEPTH)
  ) fc1_bias_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < BIAS_TENSOR_SIZE_DIM_0; j++)
    assign data_out[j] = data_vector[BIAS_PRECISION_0*j+BIAS_PRECISION_0-1:BIAS_PRECISION_0*j];

  assign data_out_valid = 1;

endmodule
