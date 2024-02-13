
/*

When FIFO receives pop, read pointer is updated directly but new data is available
on out_data when out_valid is asserted (RAM reads take 3 cycles)

*/

`timescale 1ns / 1ns

module ultraram_fifo #(
    parameter WIDTH = 512,
    parameter DEPTH = 4096
) (
    input logic core_clk,
    input logic resetn,

    input logic             push,
    input logic [WIDTH-1:0] in_data,

    input  logic             pop,
    input  logic             reset_read_ptr,
    output logic             out_valid,
    output logic [WIDTH-1:0] out_data,

    output logic [$clog2(DEPTH):0] count,
    output logic                   empty,
    output logic                   full
);

  parameter AWIDTH = $clog2(DEPTH);

  // ==================================================================================================================================================
  // Declarations
  // ==================================================================================================================================================

  logic [AWIDTH-1:0] wr_ptr;
  logic [AWIDTH-1:0] rd_ptr;
  logic [AWIDTH-1:0] read_address;

  logic pop1, pop2;

  logic wr_wrap, rd_wrap;

  // ==================================================================================================================================================
  // Logic
  // ==================================================================================================================================================

  ultraram #(
      .AWIDTH(AWIDTH),  // Address Width
      .DWIDTH(WIDTH),  // Data Width
      .NBPIPE(1)  // Number of pipeline Registers
  ) fifo (
      .core_clk(core_clk),
      .resetn  (resetn),

      .mem_en('1),  // TO DO: change for power savings

      .write_enable(push),
      .addra       (wr_ptr),
      .dina        (in_data),

      .regceb('1),            // TO DO: change for power savings
      .addrb (read_address),
      .doutb (out_data)
  );

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      wr_ptr    <= '0;
      rd_ptr    <= '0;

      pop1      <= '0;
      pop2      <= '0;
      out_valid <= '1;

      wr_wrap   <= '0;
      rd_wrap   <= '0;

      count     <= '0;
    end else begin
      wr_ptr <= push ? wr_ptr + 1'b1 : wr_ptr;

      rd_ptr <= reset_read_ptr ? '0 : pop ? rd_ptr + 1'b1 : rd_ptr;

      count  <= push && pop ? count : push ? count + 1'b1 : pop ? count - 1'b1 : count;

      // Latch out_valid to 0 when pop or to 1, 3 cycles later
      // This accounts for RAM delay
      if (pop) out_valid <= '0;
      else if (pop2) out_valid <= '1;

      pop1 <= pop;
      pop2 <= pop1;

      if (wr_ptr == {AWIDTH{1'b1}} && push) wr_wrap <= !wr_wrap;
      if (rd_ptr == {AWIDTH{1'b1}} && pop) rd_wrap <= !rd_wrap;
    end
  end

  // Pre-increment read address to account for read latency
  assign read_address = pop && (rd_ptr == DEPTH - 1) ? '0  // account for wraparound
      : pop ? rd_ptr + 1'b1 : rd_ptr;

  assign empty = (wr_ptr == rd_ptr) && !(wr_wrap ^ rd_wrap);
  assign full = (wr_ptr == rd_ptr) && (wr_wrap ^ rd_wrap);

endmodule
