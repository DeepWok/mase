
//  Xilinx UltraRAM Simple Dual Port.  This code implements 
//  a parameterizable UltraRAM block with 1 Read and 1 write
//  when addra == addrb, old data will show at doutb 

`timescale 1ns / 1ns

module ultraram #(
    parameter AWIDTH = 12,  // Address Width
    parameter DWIDTH = 512,  // Data Width
    parameter NBPIPE = 1  // Number of pipeline Registers
) (
    input  wire              core_clk,      // Clock 
    input  wire              resetn,        // Reset
    input  wire              write_enable,  // Write Enable
    input  wire              regceb,        // Output Register Enable
    input  wire              mem_en,        // Memory Enable
    input  wire [DWIDTH-1:0] dina,          // Data <wire_or_reg>  
    input  wire [AWIDTH-1:0] addra,         // Write Address
    input  wire [AWIDTH-1:0] addrb,         // Read  Address
    output reg  [DWIDTH-1:0] doutb          // Data Output
);

`ifndef FORMAL
  (* ram_style = "ultra" *)
`endif

  reg     [DWIDTH-1:0] mem            [(1<<AWIDTH)-1:0];  // Memory Declaration
  reg     [DWIDTH-1:0] memreg;
  reg     [DWIDTH-1:0] mem_pipe_reg   [     NBPIPE-1:0];  // Pipelines for memory
  reg                  mem_en_pipe_reg[       NBPIPE:0];  // Pipelines for memory enable  

  integer              i;

  // RAM : Both READ and WRITE have a latency of one
  always @(posedge core_clk) begin
    if (mem_en) begin
      if (write_enable) begin
        mem[addra] <= dina;
      end

      memreg <= mem[addrb];
    end
  end

  // The enable of the RAM goes through a pipeline to produce a
  // series of pipelined enable signals required to control the data
  // pipeline.
  always @(posedge core_clk) begin
    mem_en_pipe_reg[0] <= mem_en;
    for (i = 0; i < NBPIPE; i = i + 1) begin
      mem_en_pipe_reg[i+1] <= mem_en_pipe_reg[i];
    end
  end

  // RAM output data goes through a pipeline.
  always @(posedge core_clk) begin
    if (mem_en_pipe_reg[0]) begin
      mem_pipe_reg[0] <= memreg;
    end
  end

  always @(posedge core_clk) begin
    for (i = 0; i < NBPIPE - 1; i = i + 1) begin
      if (mem_en_pipe_reg[i+1]) begin
        mem_pipe_reg[i+1] <= mem_pipe_reg[i];
      end
    end
  end

  // Final output register gives user the option to add a reset and
  // an additional enable signal just for the data ouptut
  always @(posedge core_clk) begin
    if (!resetn) begin
      doutb <= 0;
    end else if (mem_en_pipe_reg[NBPIPE] && regceb) begin
      doutb <= mem_pipe_reg[NBPIPE-1];
    end
  end

endmodule
