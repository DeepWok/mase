`timescale 1ns / 1ps
module blk_mem_gen_0 (
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
  input logic [9:0] addra;
  input logic [7:0] dina;
  output logic [7:0] douta;

  logic [7:0] ram[0:1023];
  logic [7:0] douta_t1;
  logic [7:0] douta_t0;
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
