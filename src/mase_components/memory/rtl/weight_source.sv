
// =====================================
//     Mase Hardware
//     Parameter: fc2_weight
//     13/08/2024 16:30:42
// =====================================

`timescale 1 ns / 1 ps
module fc2_weight_rom #(
  parameter DWIDTH = 32,
  parameter MEM_SIZE = 25,
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
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
    ram[0] = 32'h00010203;
    ram[1] = 32'h04050607;
    ram[2] = 32'h0C0B0A09;
    // $readmemh("weight_rom.dat", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module fc2_weight #(
  parameter DATA_WIDTH = 32'd32,
  parameter ADDR_RANGE = 32'd25,
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  fc2_weight_rom fc2_weight_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule


`timescale 1ns / 1ps
module weight_source #(
    parameter WEIGHT_TENSOR_SIZE_DIM_0  = 16,
    parameter WEIGHT_TENSOR_SIZE_DIM_1  = 2,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 3,

    parameter WEIGHT_PARALLELISM_DIM_0 = 2,
    parameter WEIGHT_PARALLELISM_DIM_1 = 2,
    parameter OUT_DEPTH = WEIGHT_TENSOR_SIZE_DIM_0 / WEIGHT_PARALLELISM_DIM_0
) (
    input clk,
    input rst,

    output logic [WEIGHT_PRECISION_0-1:0] data_out      [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    output                       data_out_valid,
    input                        data_out_ready
);
  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(OUT_DEPTH);
  logic [COUNTER_WIDTH:0] counter;
  logic [WEIGHT_PRECISION_0-1:0]  buffered_data_out [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0];
  logic buffered_data_out_valid, buffered_data_out_ready;
  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (buffered_data_out_ready) begin
        if (counter == OUT_DEPTH - 1) counter <= 0;
        else counter <= counter + 1;
      end
    end

  logic ce0;
  assign ce0 = 1;
  assign buffered_data_out_valid = buffered_data_out_ready;
  unpacked_skid_buffer #(
  .DATA_WIDTH(WEIGHT_PRECISION_0),
  .IN_NUM(WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1)
  ) handshake_buffer (
    .clk,
    .rst,
    .data_in(buffered_data_out),
    .data_in_valid(buffered_data_out_valid),
    .data_in_ready(buffered_data_out_ready),
    .data_out,
    .data_out_valid,
    .data_out_ready
);

  logic [WEIGHT_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0-1:0] data_vector;
  fc2_weight #(
      .DATA_WIDTH(WEIGHT_PRECISION_0 * WEIGHT_TENSOR_SIZE_DIM_0),
      .ADDR_RANGE(OUT_DEPTH)
  ) fc2_weight_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1; j++)
    assign buffered_data_out[j] = data_vector[WEIGHT_PRECISION_0*j+WEIGHT_PRECISION_0-1:WEIGHT_PRECISION_0*j];


endmodule
