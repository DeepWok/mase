`timescale 1ns / 1ps
module blk_mem_gen_0 #(
    parameter DATA_WIDTH = 8,
    parameter MEM_SIZE   = 1
) (
    clka,
    ena,
    wea,
    addra,
    dina,
    douta
)
/* synthesis syn_black_box black_box_pad_pin="ena,wea[0:0],addra[9:0],dina[7:0],douta[7:0]" */
/* synthesis syn_force_seq_prim="clka" */;
  input logic clka  /* synthesis syn_isclock = 1 */;
  input logic ena;
  input logic [0:0] wea;
  input logic [$clog2(MEM_SIZE):0] addra;
  input logic [DATA_WIDTH - 1:0] dina;
  output logic [DATA_WIDTH - 1:0] douta;

  logic [DATA_WIDTH - 1:0] ram[0:MEM_SIZE-1];
  logic [DATA_WIDTH - 1:0] douta_t1;
  logic [DATA_WIDTH - 1:0] douta_t0;
  assign douta = douta_t1;

  /* verilator lint_off WIDTH */
  always_ff @(posedge clka) begin
    if (ena) begin
      if (wea) ram[addra] <= dina;
      douta_t0 <= ram[addra];
    end
    if (ena) douta_t1 <= douta_t0;
  end
endmodule
