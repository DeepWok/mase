
/*

When FIFO receives pop, read pointer is updated directly but new data is available
on out_data when out_valid is asserted (RAM reads take 3 cycles)

*/

// TO DO: feature count will break if writing and reading at the same time

module bram_fifo #(
    parameter WRITE_WIDTH = 64,
    parameter WRITE_DEPTH = 512,
    parameter READ_WIDTH  = 32,
    parameter READ_DEPTH  = 1024,
    parameter BRAM_TYPE   = 0
) (
    input logic core_clk,
    input logic resetn,

    input logic                   push,
    input logic [WRITE_WIDTH-1:0] in_data,

    input  logic                  pop,
    output logic                  out_valid,
    output logic [READ_WIDTH-1:0] out_data,

    output logic [$clog2(READ_DEPTH):0] count,
    output logic                        empty,
    output logic                        full
);

  parameter WRITE_ADDR_WIDTH = $clog2(WRITE_DEPTH);
  parameter READ_ADDR_WIDTH = $clog2(READ_DEPTH);

  // ==================================================================================================================================================
  // Declarations
  // ==================================================================================================================================================

  logic [WRITE_ADDR_WIDTH-1:0] wr_ptr;
  logic [ READ_ADDR_WIDTH-1:0] rd_ptr;
  logic [ READ_ADDR_WIDTH-1:0] read_address;
  logic                        pop_q;

  // Pre-increment read address to account for read latency
  assign read_address = pop && (rd_ptr == READ_DEPTH - 1) ? '0  // account for wraparound
      : pop ? rd_ptr + 1'b1 : rd_ptr;

  logic wr_wrap, rd_wrap;

  // ==================================================================================================================================================
  // Logic
  // ==================================================================================================================================================

  if (BRAM_TYPE == 0) begin

    scale_factor_queue scale_factor_queue_i (
        .clka (core_clk),
        .ena  (1'b1),
        .wea  (push),
        .addra(wr_ptr),
        .dina (in_data),

        .clkb (core_clk),
        .enb  (1'b1),
        .addrb(read_address),
        .doutb(out_data),

        .sleep(1'b0)
    );

  end else begin
    assign out_data = '0;
  end

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      wr_ptr    <= '0;
      rd_ptr    <= '0;

      pop_q     <= '0;
      out_valid <= '1;

      wr_wrap   <= '0;
      rd_wrap   <= '0;

      count     <= '0;
    end else begin
      if (push) begin
        wr_ptr <= wr_ptr == WRITE_DEPTH - 1 ? '0 : wr_ptr + 1'b1;
      end

      if (pop) begin
        rd_ptr <= rd_ptr == READ_DEPTH - 1 ? '0 : rd_ptr + 1'b1;
      end

      count <= push && pop ? count : push ? count + 1'b1 : pop ? count - 1'b1 : count;

      // Latch out_valid to 0 when pop or to 1, 3 cycles later
      // This accounts for RAM delay
      out_valid <= pop ? '0 : pop_q ? '1 : out_valid;

      pop_q <= pop;

      if ((wr_ptr == (WRITE_DEPTH - 1)) && push) wr_wrap <= !wr_wrap;
      if ((rd_ptr == (READ_DEPTH - 1)) && pop) rd_wrap <= !rd_wrap;
    end
  end

  assign empty = (count == '0);
  assign full  = (count == '1);

endmodule
