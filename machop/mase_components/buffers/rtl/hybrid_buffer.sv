
// Hybrid buffer used as intermediate storage between AGE -> FTE and FTE -> MPE
// Write interface behaves as RAM (write_enable, write_address)
// Read interface behaves as FIFO

module hybrid_buffer #(
    parameter NUM_SLOTS = 16,
    parameter WRITE_WIDTH = 64,
    parameter WRITE_DEPTH = 512,
    parameter READ_WIDTH = 32,
    parameter READ_DEPTH = 1024,
    parameter BUFFER_TYPE = "AGGREGATION",
    parameter SLOT_ID_WIDTH = 20
) (
    input logic core_clk,
    input logic resetn,

    input logic [NUM_SLOTS-1:0]                    set_node_id_valid,
    input logic [NUM_SLOTS-1:0][SLOT_ID_WIDTH-1:0] set_node_id,

    output logic [NUM_SLOTS-1:0][SLOT_ID_WIDTH-1:0] slot_node_id,

    input logic [NUM_SLOTS-1:0]                          write_enable,
    input logic [NUM_SLOTS-1:0][$clog2(WRITE_DEPTH)-1:0] write_address,
    input logic [NUM_SLOTS-1:0][        WRITE_WIDTH-1:0] write_data,

    input  logic [NUM_SLOTS-1:0]                 pop,
    output logic [NUM_SLOTS-1:0]                 out_feature_valid,
    output logic [NUM_SLOTS-1:0][READ_WIDTH-1:0] out_feature,

    output logic [NUM_SLOTS-1:0][$clog2(READ_DEPTH)-1:0] feature_count,
    output logic [NUM_SLOTS-1:0]                         slot_free
);

  for (genvar slot = 0; slot < NUM_SLOTS; slot++) begin
    hybrid_buffer_slot #(
        .WRITE_WIDTH(WRITE_WIDTH),
        .WRITE_DEPTH(WRITE_DEPTH),
        .READ_WIDTH (READ_WIDTH),
        .READ_DEPTH (READ_DEPTH),
        .BUFFER_TYPE(BUFFER_TYPE)
    ) slot_i (
        .core_clk(core_clk),
        .resetn  (resetn),

        .write_enable (write_enable[slot]),
        .write_address(write_address[slot]),
        .write_data   (write_data[slot]),

        .pop              (pop[slot]),
        .out_feature_valid(out_feature_valid[slot]),
        .out_feature      (out_feature[slot]),

        .feature_count(feature_count[slot]),
        .slot_free    (slot_free[slot])
    );

    // Node IDs
    always_ff @(posedge core_clk or negedge resetn) begin
      if (!resetn) begin
        slot_node_id[slot] <= 0;

      end else if (set_node_id && set_node_id_valid[slot]) begin
        slot_node_id[slot] <= set_node_id[slot];
      end
    end
  end

endmodule
