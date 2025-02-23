
// =====================================
//     Mase Hardware
//     Parameter: attention_query_bias
//     23/02/2025 14:08:25
// =====================================

`timescale 1 ns / 1 ps
module attention_query_bias_rom #(
  parameter DWIDTH = 104,
  parameter MEM_SIZE = 12,
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
    $readmemb("mxint_vit_block/hardware/rtl/attention_query_bias_rom.dat", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module attention_query_bias #(
  parameter DATA_WIDTH = 32'd104,
  parameter ADDR_RANGE = 32'd12,
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  attention_query_bias_rom attention_query_bias_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule


`timescale 1ns / 1ps
module attention_query_bias_source #(
    parameter QUERY_BIAS_TENSOR_SIZE_DIM_0  = -1,
    parameter QUERY_BIAS_TENSOR_SIZE_DIM_1  = -1,
    parameter QUERY_BIAS_PRECISION_0 = -1,
    parameter QUERY_BIAS_PRECISION_1 = -1,

    parameter QUERY_BIAS_PARALLELISM_DIM_0 = -1,
    parameter QUERY_BIAS_PARALLELISM_DIM_1 = -1,
    parameter OUT_DEPTH = (QUERY_BIAS_TENSOR_SIZE_DIM_0 / QUERY_BIAS_PARALLELISM_DIM_0) * (QUERY_BIAS_TENSOR_SIZE_DIM_1 / QUERY_BIAS_PARALLELISM_DIM_1)
) (
    input clk,
    input rst,

    output logic [QUERY_BIAS_PRECISION_0-1:0] mdata_out      [QUERY_BIAS_PARALLELISM_DIM_0 * QUERY_BIAS_PARALLELISM_DIM_1-1:0],
    output logic [QUERY_BIAS_PRECISION_1-1:0] edata_out,
    output                       data_out_valid,
    input                        data_out_ready
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
  logic [1:0] clear;
  always_ff @(posedge clk)
    if (rst) clear <= 0;
    else if ((data_out_ready == 1) && (clear != 2)) clear <= clear + 1;
  logic ce0;
  assign ce0 = data_out_ready;

  localparam TOTAL_WIDTH = QUERY_BIAS_PRECISION_0*(QUERY_BIAS_PARALLELISM_DIM_0*QUERY_BIAS_PARALLELISM_DIM_1) + QUERY_BIAS_PRECISION_1;
  logic [TOTAL_WIDTH-1:0] data_vector;
  attention_query_bias #(
      .DATA_WIDTH(TOTAL_WIDTH),
      .ADDR_RANGE(OUT_DEPTH)
  ) attention_query_bias_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < QUERY_BIAS_PARALLELISM_DIM_0 * QUERY_BIAS_PARALLELISM_DIM_1; j++)
    assign mdata_out[j] = data_vector[QUERY_BIAS_PRECISION_0*j+QUERY_BIAS_PRECISION_0-1 + QUERY_BIAS_PRECISION_1:QUERY_BIAS_PRECISION_0*j + QUERY_BIAS_PRECISION_1];
  assign edata_out = data_vector[QUERY_BIAS_PRECISION_1-1 : 0];
  assign data_out_valid = clear == 2;

endmodule
