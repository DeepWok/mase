
module hybrid_buffer_slot #(
    parameter WRITE_WIDTH = 64,
    parameter WRITE_DEPTH = 512,
    parameter READ_WIDTH  = 32,
    parameter READ_DEPTH  = 1024,
    parameter BUFFER_TYPE = "AGGREGATION"
) (
    input logic core_clk,
    input logic resetn,

    input logic                           write_enable,
    input logic [$clog2(WRITE_DEPTH)-1:0] write_address,
    input logic [        WRITE_WIDTH-1:0] write_data,

    input  logic                  pop,
    output logic                  out_feature_valid,
    output logic [READ_WIDTH-1:0] out_feature,

    output logic [$clog2(READ_DEPTH)-1:0] feature_count,
    output logic                          slot_free
);

  logic [$clog2(READ_DEPTH)-1:0] rd_ptr;
  logic [$clog2(READ_DEPTH)-1:0] read_address;
  logic                          pop_q;

  // Pre-increment read address to account for read latency
  assign read_address = pop && (rd_ptr == READ_DEPTH - 1) ? '0  // account for wraparound
      : pop ? rd_ptr + 1'b1 : rd_ptr;

  // Instances
  // ------------------------------------------------------------

  if (BUFFER_TYPE == "AGGREGATION") begin

    aggregation_buffer_sdp_bram fifo (
        .clka (core_clk),       // input wire clka
        .ena  ('1),             // input wire ena
        .wea  (write_enable),   // input wire [0 : 0] wea
        .addra(write_address),  // input wire [8 : 0] addra
        .dina (write_data),     // input wire [63 : 0] dina

        .clkb (core_clk),      // input wire clkb
        .enb  ('1),            // input wire enb
        .addrb(read_address),  // input wire [9 : 0] addrb
        .doutb(out_feature),   // output wire [31 : 0] doutb

        .sleep('0)  // input wire sleep
    );

  end else if (BUFFER_TYPE == "TRANSFORMATION") begin

    transformation_buffer_sdp_bram fifo (
        .clka (core_clk),       // input wire clka
        .ena  ('1),             // input wire ena
        .wea  (write_enable),   // input wire [0 : 0] wea
        .addra(write_address),  // input wire [5 : 0] addra
        .dina (write_data),     // input wire [511 : 0] dina

        .clkb (core_clk),      // input wire clkb
        .enb  ('1),            // input wire enb
        .addrb(read_address),  // input wire [9 : 0] addrb
        .doutb(out_feature)    // output wire [31 : 0] doutb
    );

  end

  // Logic
  // ------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      rd_ptr            <= '0;
      feature_count     <= 0;
      out_feature_valid <= '1;
      pop_q             <= '0;

    end else begin
      if (write_enable) begin
        feature_count <= feature_count + 'd2;
      end

      // Latch out_valid to 0 when pop or to 1, 3 cycles later
      // This accounts for RAM delay
      if (pop) begin
        rd_ptr <= rd_ptr + 1;
        feature_count <= feature_count - 1'b1;
      end

      // Latch out_valid to 0 when pop or to 1, 3 cycles later
      // This accounts for RAM delay
      out_feature_valid <= pop ? '0 : pop_q ? '1 : out_feature_valid;

      pop_q <= pop;

    end
  end

  assign slot_free = (feature_count == '0);

endmodule
