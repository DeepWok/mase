
module hybrid_buffer_driver #(
    parameter BUFFER_SLOTS = 16,
    parameter MAX_PULSES_PER_SLOT = top_pkg::MAX_FEATURE_COUNT
) (
    input logic core_clk,
    input logic resetn,

    input logic begin_dump,
    input logic pulse,

    input logic [$clog2(MAX_PULSES_PER_SLOT)-1:0] pulse_limit,

    output logic [BUFFER_SLOTS-1:0] slot_pop_shift
);

  logic [$clog2(MAX_PULSES_PER_SLOT)-1:0] slot_pop_counter;

  // Shift register to flush through weights matrix diagonally
  for (genvar slot = 1; slot < BUFFER_SLOTS; slot++) begin
    always_ff @(posedge core_clk or negedge resetn) begin
      if (!resetn) begin
        slot_pop_shift[slot] <= '0;

        // Clear shift register when starting new weight dump
      end else if (begin_dump) begin
        slot_pop_shift[slot] <= '0;

        // Shift register when accepting weight channel response
      end else if (pulse) begin
        slot_pop_shift[slot] <= slot_pop_shift[slot-1];
      end
    end
  end

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      slot_pop_shift[0] <= '0;  // Head of shift register
      slot_pop_counter  <= '0;

      // Starting new feature dump, reset all flags and counters
    end else if (begin_dump) begin
      slot_pop_shift[0] <= '1;
      slot_pop_counter  <= '0;

    end else if (pulse) begin
      // Increment when popping any slots, but latch at '1
      slot_pop_counter <= (slot_pop_counter == (pulse_limit-1)) ? slot_pop_counter : (slot_pop_counter + 1'b1);

      // If accepting weight channel response, new data is available on all slot FIFOs so shift register
      slot_pop_shift[0] <= (slot_pop_counter != (pulse_limit - 1));

    end
  end

endmodule
