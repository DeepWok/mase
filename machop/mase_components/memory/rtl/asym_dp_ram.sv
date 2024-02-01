
module asym_dp_ram #(
    parameter WRITE_WIDTH = 64,
    parameter WRITE_DEPTH = 512,
    parameter READ_WIDTH  = 32,
    parameter READ_DEPTH  = 1024
) (
    input logic core_clk,
    input logic resetn,

    input logic wea,
    input logic [$clog2(WRITE_DEPTH)-1:0] addr_a,
    input logic [WRITE_WIDTH-1:0] in_data_a,

    input logic [$clog2(READ_DEPTH)-1:0] addr_b,
    output logic [READ_WIDTH-1:0] out_data_b
);

  parameter INNER_BLOCKS = WRITE_WIDTH / READ_WIDTH;
  parameter INNER_BLOCKS_WIDTH = $clog2(INNER_BLOCKS);

  logic [$clog2(READ_DEPTH)-1:0][READ_WIDTH-1:0] mem;
  logic [$clog2(READ_DEPTH)-1:0] base_addr;
  logic [$clog2(READ_DEPTH)-1:0] addr_b_q;

  assign base_addr = {addr_a, {INNER_BLOCKS_WIDTH{1'b0}}};

  // Write logic
  // ----------------------------------------------------------------------------

  for (genvar loc = 0; loc < READ_DEPTH; loc++) begin

    parameter OFFSET = loc / INNER_BLOCKS;

    always_ff @(posedge core_clk or negedge resetn) begin

      if (!resetn) begin
        mem[loc] <= '0;

      end else begin

        if (wea && (loc[$clog2(READ_DEPTH)-1:INNER_BLOCKS_WIDTH] == addr_a)) begin
          mem[loc] <= in_data_a[(OFFSET+1)*READ_WIDTH : OFFSET*READ_WIDTH];
        end
      end

    end
  end

  // Read logic
  // ----------------------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin

    if (!resetn) begin
      addr_b_q   <= '0;
      out_data_b <= '0;

    end else begin

      // Register read address
      addr_b_q   <= addr_b;

      // Register output data
      out_data_b <= mem[addr_b_q];
    end
  end

endmodule
