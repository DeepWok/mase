mxint_template = """
// =====================================
//     Mase Hardware
//     Parameter: {node_param_name}
//     {date_time}
// =====================================

`timescale 1 ns / 1 ps
module {node_param_name}_mantissa_rom #(
  parameter DWIDTH = {m_width},
  parameter MEM_SIZE = {m_mem_size},
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
) (
    input                       clk,
    input  logic [AWIDTH-1:0]   addr0,
    input                       ce0,
    output logic [DWIDTH-1:0]   q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemb("{filename}_block.dat", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];
endmodule

`timescale 1 ns / 1 ps
module {node_param_name}_exponent_rom #(
  parameter DWIDTH = {e_width},
  parameter MEM_SIZE = {e_mem_size},
  parameter AWIDTH = $clog2(MEM_SIZE) + 1
) (
    input                       clk,
    input  logic [AWIDTH-1:0]   addr0,
    input                       ce0,
    output logic [DWIDTH-1:0]   q0
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;

  initial begin
    $readmemh("{filename}_exp.dat", ram);
  end

  assign q0 = q0_t1;

  always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
  always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];
endmodule

`timescale 1ns / 1ps
module {node_param_name}_source #(
    parameter {verilog_param_name}_TENSOR_SIZE_DIM_1  = 1,
    parameter {verilog_param_name}_TENSOR_SIZE_DIM_0  = 32,
    parameter {verilog_param_name}_PRECISION_0 = 16,
    parameter {verilog_param_name}_PRECISION_1 = 3,

    parameter {verilog_param_name}_PARALLELISM_DIM_0 = 1,
    parameter {verilog_param_name}_PARALLELISM_DIM_1 = 1,
    parameter OUT_DEPTH = (({verilog_param_name}_TENSOR_SIZE_DIM_0 + {verilog_param_name}_PARALLELISM_DIM_0 - 1) / {verilog_param_name}_PARALLELISM_DIM_0) 
                        * (({verilog_param_name}_TENSOR_SIZE_DIM_1 + {verilog_param_name}_PARALLELISM_DIM_1 - 1) / {verilog_param_name}_PARALLELISM_DIM_1)
) (
    input clk,
    input rst,

    output logic [{verilog_param_name}_PRECISION_0-1:0] mdata_out      [{verilog_param_name}_PARALLELISM_DIM_0 * {verilog_param_name}_PARALLELISM_DIM_1-1:0],
    output logic [{verilog_param_name}_PRECISION_1-1:0] edata_out,
    output                                              data_out_valid,
    input                                               data_out_ready
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

  logic [{verilog_param_name}_PRECISION_0*{verilog_param_name}_PARALLELISM_DIM_0*{verilog_param_name}_PARALLELISM_DIM_1-1:0] data_vector;
  {node_param_name}_mantissa_rom #() {node_param_name}_mantissa (
      .clk(clk),
      .addr0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  {node_param_name}_exponent_rom #() {node_param_name}_exponent (
      .clk(clk),
      .addr0(counter),
      .ce0(ce0),
      .q0(edata_out)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < {verilog_param_name}_PARALLELISM_DIM_0 * {verilog_param_name}_PARALLELISM_DIM_1; j++)
    assign mdata_out[j] = data_vector[{verilog_param_name}_PRECISION_0*(j+1)-1:{verilog_param_name}_PRECISION_0*j];

  assign data_out_valid = clear == 2;

endmodule
"""
