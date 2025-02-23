
// =====================================
//     Mase Hardware
//     Parameter: attention_value_weight
//     23/02/2025 13:37:22
// =====================================

`timescale 1 ns / 1 ps
module attention_value_weight_rom #(
  parameter DWIDTH = 1544,
  parameter MEM_SIZE = 16,
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
    $readmemb("sim_build/attention_value_weight_rom.dat", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

endmodule

`timescale 1 ns / 1 ps
module attention_value_weight #(
  parameter DATA_WIDTH = 32'd1544,
  parameter ADDR_RANGE = 32'd16,
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  attention_value_weight_rom attention_value_weight_rom_U (
      .clk(clk),
      .addr0(address0),
      .ce0(ce0),
      .q0(q0)
  );

endmodule


`timescale 1ns / 1ps
module attention_value_weight_source #(
    parameter VALUE_WEIGHT_TENSOR_SIZE_DIM_0  = -1,
    parameter VALUE_WEIGHT_TENSOR_SIZE_DIM_1  = -1,
    parameter VALUE_WEIGHT_PRECISION_0 = -1,
    parameter VALUE_WEIGHT_PRECISION_1 = -1,

    parameter VALUE_WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter VALUE_WEIGHT_PARALLELISM_DIM_1 = -1,
    parameter OUT_DEPTH = (VALUE_WEIGHT_TENSOR_SIZE_DIM_0 / VALUE_WEIGHT_PARALLELISM_DIM_0) * (VALUE_WEIGHT_TENSOR_SIZE_DIM_1 / VALUE_WEIGHT_PARALLELISM_DIM_1)
) (
    input clk,
    input rst,

    output logic [VALUE_WEIGHT_PRECISION_0-1:0] mdata_out      [VALUE_WEIGHT_PARALLELISM_DIM_0 * VALUE_WEIGHT_PARALLELISM_DIM_1-1:0],
    output logic [VALUE_WEIGHT_PRECISION_1-1:0] edata_out,
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

  localparam TOTAL_WIDTH = VALUE_WEIGHT_PRECISION_0*(VALUE_WEIGHT_PARALLELISM_DIM_0*VALUE_WEIGHT_PARALLELISM_DIM_1) + VALUE_WEIGHT_PRECISION_1;
  logic [TOTAL_WIDTH-1:0] data_vector;
  attention_value_weight #(
      .DATA_WIDTH(TOTAL_WIDTH),
      .ADDR_RANGE(OUT_DEPTH)
  ) attention_value_weight_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < VALUE_WEIGHT_PARALLELISM_DIM_0 * VALUE_WEIGHT_PARALLELISM_DIM_1; j++)
    assign mdata_out[j] = data_vector[VALUE_WEIGHT_PRECISION_0*j+VALUE_WEIGHT_PRECISION_0-1 + VALUE_WEIGHT_PRECISION_1:VALUE_WEIGHT_PRECISION_0*j + VALUE_WEIGHT_PRECISION_1];
  assign edata_out = data_vector[VALUE_WEIGHT_PRECISION_1-1 : 0];
  assign data_out_valid = clear == 2;

endmodule
