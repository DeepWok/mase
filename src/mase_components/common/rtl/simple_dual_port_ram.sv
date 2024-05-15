`timescale 1ns / 1ps

module simple_dual_port_ram #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 9,
    parameter SIZE       = 512,

    // Ram style can be:
    // "block": BRAM
    // "distributed": LUTRAM
    // "registers"
    // "ultra": URAM on Ultrascale+
    // "mixed": combination of above to fill in space
    // "auto": let synthesis choose
    /* verilator lint_off UNUSEDPARAM */
    parameter RAM_STYLE = "auto"
) (
    input logic clk,

    // Write Port
    input logic [DATA_WIDTH-1:0] wr_din,
    input logic [ADDR_WIDTH-1:0] wr_addr,
    input logic                  wr_en,

    // Read Port
    input  logic [ADDR_WIDTH-1:0] rd_addr,
    output logic [DATA_WIDTH-1:0] rd_dout
);

  (* ram_style = RAM_STYLE *)
  logic [DATA_WIDTH-1:0] mem[SIZE-1:0];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_din;
    end
    rd_dout <= mem[rd_addr];
  end

endmodule
