
/*
Receives number of bytes needed
Use transaction size of 64 bytes (ARSIZE = 0x6) to saturate 512b data bus
Calculate required burst length and number of transactions
When crossing a 4 kb address boundary, issue several transactions with varying ID

Currently issuing all read transactions before asserting rready to receive read responses
This may be bottleneck - in the future, implement two separate state machines

*/

module axi_read_master #(
    parameter MAX_BYTE_COUNT = 1000000000,
    parameter AXI_ADDRESS_WIDTH = 34,
    parameter DATA_WIDTH = 512
) (
    input logic core_clk,
    input logic resetn,

    // Request interface
    input  logic                              fetch_req_valid,
    output logic                              fetch_req_ready,
    input  logic [     AXI_ADDRESS_WIDTH-1:0] fetch_start_address,
    input  logic [$clog2(MAX_BYTE_COUNT)-1:0] fetch_byte_count,

    // Response interface
    output logic                  fetch_resp_valid,
    input  logic                  fetch_resp_ready,
    output logic                  fetch_resp_last,
    output logic [DATA_WIDTH-1:0] fetch_resp_data,
    output logic [           3:0] fetch_resp_axi_id,

    // AXI Read-Only Interface
    output logic [AXI_ADDRESS_WIDTH-1:0] axi_araddr,
    output logic [                  1:0] axi_arburst,
    output logic [                  3:0] axi_arcache,
    output logic [                  3:0] axi_arid,
    output logic [                  7:0] axi_arlen,
    output logic [                  0:0] axi_arlock,
    output logic [                  2:0] axi_arprot,
    output logic [                  3:0] axi_arqos,
    output logic [                  2:0] axi_arsize,
    output logic                         axi_arvalid,
    input  logic                         axi_arready,
    input  logic [       DATA_WIDTH-1:0] axi_rdata,
    input  logic [                  3:0] axi_rid,
    input  logic                         axi_rlast,
    input  logic                         axi_rvalid,
    output logic                         axi_rready,
    input  logic [                  1:0] axi_rresp
);

  // (WIDTH_X - 1)/WIDTH_Y + 1 = WIDTH_X/WIDTH_Y (rounded up)
  parameter MAX_TOTAL_BEATS = (MAX_BYTE_COUNT - 1) / 64 + 1;  // 64
  parameter MAX_TRANSACTIONS = (MAX_TOTAL_BEATS - 1) / 256 + 1;  // 1 

  typedef enum logic [2:0] {
    IDLE = 3'd0,
    AR   = 3'd1,
    R    = 3'd2,
    RESP = 3'd3
  } fetch_fsm_e;

  // ==================================================================================================================================================
  // Declarations
  // ==================================================================================================================================================

  fetch_fsm_e fetch_state, fetch_state_n;

  logic [  $clog2(MAX_BYTE_COUNT)-1:0] req_fetch_byte_count;

  logic                                accepting_fetch_request;
  logic                                accepting_axi_read_transaction;
  logic                                accepting_axi_read_response;

  logic [ $clog2(MAX_TOTAL_BEATS)-1:0] required_beat_count;
  logic [                         7:0] burst_length_final_transaction;
  logic [$clog2(MAX_TRANSACTIONS)-1:0] required_transaction_count;

  logic [$clog2(MAX_TRANSACTIONS)-1:0] sent_transactions;
  logic [ $clog2(MAX_TOTAL_BEATS)-1:0] beats_requested;
  logic [ $clog2(MAX_TOTAL_BEATS)-1:0] beats_received;
  logic [                         7:0] received_read_responses;

  logic [       AXI_ADDRESS_WIDTH-1:0] current_transaction_address;

  logic                                last_transaction_pending;
  logic                                last_read_response_pending;


  // ==================================================================================================================================================
  // Logic
  // ==================================================================================================================================================

  always_comb begin

    accepting_fetch_request = fetch_req_valid && fetch_req_ready;
    accepting_axi_read_transaction = axi_arvalid && axi_arready;
    accepting_axi_read_response = axi_rvalid && axi_rready;

    // byte_count / 64 since the data bus is 512b - first divide by 64 rounding down by taking bits [MAX:6], then add 1 if there is a remainder
    required_beat_count = req_fetch_byte_count[$clog2(MAX_BYTE_COUNT)-1:6] +
        (req_fetch_byte_count[5:0] == '0 ? 1'b0 : 1'b1);


    // Max burst length per transaction is 64 such as to not cross a 4kb address boundary, so take modulus 64
    burst_length_final_transaction = required_beat_count[5:0];
    required_transaction_count = required_beat_count[$clog2(MAX_TOTAL_BEATS)-1:6] +
        ((burst_length_final_transaction == '0) ? 1'b0 : 1'b1);

    last_transaction_pending = sent_transactions == (required_transaction_count - 1'b1);
    last_read_response_pending = beats_received == (required_beat_count - 1'b1);
  end

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      fetch_state                 <= IDLE;

      req_fetch_byte_count        <= '0;
      current_transaction_address <= '0;

      sent_transactions           <= '0;
      beats_requested             <= '0;

      received_read_responses     <= '0;
      beats_received              <= '0;

    end else if (fetch_state_n == IDLE) begin
      fetch_state                 <= fetch_state_n;

      req_fetch_byte_count        <= '0;
      current_transaction_address <= '0;

      sent_transactions           <= '0;
      beats_requested             <= '0;

      received_read_responses     <= '0;
      beats_received              <= '0;

    end else begin
      fetch_state <= fetch_state_n;

      if (accepting_fetch_request) begin
        req_fetch_byte_count <= fetch_byte_count;
        current_transaction_address     <= fetch_start_address; // gets incremented for subsequent transactions
      end

      // Accepting Read
      if (fetch_state == AR && accepting_axi_read_transaction) begin
        sent_transactions <= sent_transactions + 1'b1;
        beats_requested <= beats_requested + axi_arlen;
        current_transaction_address <= current_transaction_address + 13'd4096; // increment address by 64 beats * 64 bytes
      end

      // Accepting Read response
      if (fetch_state == R && accepting_axi_read_response) begin
        received_read_responses <= received_read_responses + 1'b1;
        beats_received          <= beats_received + 1'b1;
      end
    end
  end

  // State Machine
  // -----------------------------------------

  always_comb begin
    fetch_state_n = fetch_state;

    case (fetch_state)
      IDLE: fetch_state_n = accepting_fetch_request ? AR : IDLE;
      AR: fetch_state_n = (accepting_axi_read_transaction && last_transaction_pending) ? R : AR;
      R: fetch_state_n = (accepting_axi_read_response && last_read_response_pending) ? (IDLE) : R;
    endcase
  end

  // AXI Signals
  // ----------------------------------------------

  always_comb begin
    axi_arvalid = (fetch_state == AR);
    axi_araddr = current_transaction_address;
    // axi_arid        = sent_transactions[3:0];
    axi_arid = '0;

    // use FIXED encoding for when a single burst is enough, otherwise INCR
    axi_arburst = (req_fetch_byte_count > 64) ? 2'b01 : '0;
    axi_arsize = 3'b110;  // = 64 bytes
    // request up to 64 bursts of 64 bytes such as to not cross 4kb address boundary
    axi_arlen       = last_transaction_pending && (burst_length_final_transaction == '0) ? 8'd63 // +1 = 64
    : last_transaction_pending && !(burst_length_final_transaction == '0) ? burst_length_final_transaction - 1
                    : 8'd63; // +1 = 64

    // Unused
    axi_arcache = '0;
    axi_arlock = '0;
    axi_arprot = '0;
    axi_arqos = '0;

    // How to handle responses?
    axi_rready = (fetch_state == R) && fetch_resp_ready;
  end

  // Fetch REQ/RESP interface
  // ----------------------------------------------

  always_comb begin
    fetch_req_ready   = (fetch_state == IDLE);

    // Fetch Responses
    fetch_resp_valid  = (fetch_state == R) && axi_rvalid;
    fetch_resp_last   = (fetch_state == R) && axi_rvalid && axi_rlast;
    fetch_resp_data   = axi_rdata;
    fetch_resp_axi_id = axi_rid;
  end

endmodule
