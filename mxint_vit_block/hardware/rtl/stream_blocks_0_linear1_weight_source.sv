
// =====================================
//     Mase Hardware
//     Parameter: stream_blocks_0_linear1_weight
//     14/01/2025 01:42:18
// =====================================

`timescale 1 ns / 1 ps
// module stream_blocks_0_linear1_weight_rom #(
//   parameter DWIDTH = 196,
//   parameter MEM_SIZE = 3072,
//   parameter AWIDTH = $clog2(MEM_SIZE) + 1
// ) (
//   input clk,
//   input logic [AWIDTH-1:0] addr0,
//   input ce0,
//   output logic [DWIDTH-1:0] q0
// );

//  (* ram_style = "ultra" *) logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
//   logic [DWIDTH-1:0] q0_t0;
//   logic [DWIDTH-1:0] q0_t1;

//   initial begin
//     $readmemb("mxint_vit_block/hardware/rtl/stream_blocks_0_linear1_weight_rom.dat", ram);
//   end

//   assign q0 = q0_t1;

//   always_ff @(posedge clk) if (ce0) q0_t1 <= q0_t0;
//   always_ff @(posedge clk) if (ce0) q0_t0 <= ram[addr0];

// endmodule

module stream_blocks_0_linear1_weight_rom #(
  parameter DWIDTH = 196,  // Data Width
  parameter MEM_SIZE = 3072,
  parameter AWIDTH = $clog2(MEM_SIZE) + 1,  // Address Width
  parameter NBPIPE = 3    // Number of pipeline Registers
 ) ( 
    input clk,                    // Clock 
    input rst,                    // Reset
    input we,                     // Write Enable
    input regce,                  // Output Register Enable
    input mem_en,                 // Memory Enable
    input [DWIDTH-1:0] din,       // Data Input  
    input [AWIDTH-1:0] addr,      // Address Input
    output reg [DWIDTH-1:0] dout  // Data Output
   );
(* ram_style = "ultra" *)
reg [DWIDTH-1:0] mem[0:MEM_SIZE-1];        // Memory Declaration
reg [DWIDTH-1:0] memreg;              
reg [DWIDTH-1:0] mem_pipe_reg[NBPIPE-1:0];    // Pipelines for memory
reg mem_en_pipe_reg[NBPIPE:0];                // Pipelines for memory enable  

integer          i;
// initial begin
// $readmemb("mxint_vit_block/hardware/rtl/stream_blocks_0_linear1_weight_rom.dat", mem);
// end
// RAM : Both READ and WRITE have a latency of one
always @ (posedge clk)
begin
 if(mem_en) 
  begin
  if(we)
    mem[addr] <= din;
  else
   memreg <= mem[addr];
  end     
end

// The enable of the RAM goes through a pipeline to produce a
// series of pipelined enable signals required to control the data
// pipeline.
always @ (posedge clk)
begin
 mem_en_pipe_reg[0] <= mem_en;
 for (i=0; i<NBPIPE; i=i+1)
   mem_en_pipe_reg[i+1] <= mem_en_pipe_reg[i];
end

// RAM output data goes through a pipeline.
always @ (posedge clk)
begin
 if (mem_en_pipe_reg[0])
  mem_pipe_reg[0] <= memreg;
end  
   
always @ (posedge clk)
begin
  for (i = 0; i < NBPIPE-1; i = i+1)
   if (mem_en_pipe_reg[i+1])
     mem_pipe_reg[i+1] <= mem_pipe_reg[i];
end         
   
// Final output register gives user the option to add a reset and
// an additional enable signal just for the data ouptut
always @ (posedge clk)
begin
 if (rst)
  dout <= 0;
 else if (mem_en_pipe_reg[NBPIPE] && regce)
  dout <= mem_pipe_reg[NBPIPE-1];
end   
endmodule

`timescale 1 ns / 1 ps
module stream_blocks_0_linear1_weight #(
  parameter DATA_WIDTH = 32'd196,
  parameter ADDR_RANGE = 32'd3072,
  parameter ADDR_WIDTH = $clog2(ADDR_RANGE) + 1
) (
  input reset,
  input clk,
  input logic [ADDR_WIDTH - 1:0] address0,
  input ce0,
  output logic [DATA_WIDTH - 1:0] q0
);

  stream_blocks_0_linear1_weight_rom stream_blocks_0_linear1_weight_rom_U (
      .clk(clk),
      .rst(reset),
      .we(1),
      .din(0),
      .regce(ce0),
      .mem_en(ce0),
      .addr(address0),
      .dout(q0)
  );

endmodule


`timescale 1ns / 1ps
module stream_blocks_0_linear1_weight_source #(
    parameter WEIGHT_TENSOR_SIZE_DIM_0  = -1,
    parameter WEIGHT_TENSOR_SIZE_DIM_1  = -1,
    parameter WEIGHT_PRECISION_0 = -1,
    parameter WEIGHT_PRECISION_1 = -1,

    parameter WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter WEIGHT_PARALLELISM_DIM_1 = -1,
    parameter OUT_DEPTH = (WEIGHT_TENSOR_SIZE_DIM_0 / WEIGHT_PARALLELISM_DIM_0) * (WEIGHT_TENSOR_SIZE_DIM_1 / WEIGHT_PARALLELISM_DIM_1)
) (
    input clk,
    input rst,

    output logic [WEIGHT_PRECISION_0-1:0] mdata_out      [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    output logic [WEIGHT_PRECISION_1-1:0] edata_out,
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

  localparam TOTAL_WIDTH = WEIGHT_PRECISION_0*(WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1) + WEIGHT_PRECISION_1;
  logic [TOTAL_WIDTH-1:0] data_vector;
  stream_blocks_0_linear1_weight #(
      .DATA_WIDTH(TOTAL_WIDTH),
      .ADDR_RANGE(OUT_DEPTH)
  ) stream_blocks_0_linear1_weight_mem (
      .clk(clk),
      .reset(rst),
      .address0(counter),
      .ce0(ce0),
      .q0(data_vector)
  );

  // Cocotb/verilator does not support array flattening, so
  // we need to manually add some reshaping process.
  for (genvar j = 0; j < WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1; j++)
    assign mdata_out[j] = data_vector[WEIGHT_PRECISION_0*j+WEIGHT_PRECISION_0-1 + WEIGHT_PRECISION_1:WEIGHT_PRECISION_0*j + WEIGHT_PRECISION_1];
  assign edata_out = data_vector[WEIGHT_PRECISION_1-1 : 0];
  assign data_out_valid = clear == 2;

endmodule
