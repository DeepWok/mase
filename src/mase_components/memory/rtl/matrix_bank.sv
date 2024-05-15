/*

Features to implement:
- Resetting after layer finished. Accept request while in FEATURES_WAITING,
  clear all counters and flags and transition to REQ_FETCH

*/

import matrix_bank_pkg::*;

module matrix_bank #(
    parameter PRECISION = 0, // 0 = FP32, 1 = FP16
    parameter DATA_WIDTH = 32,
    parameter AXI_ADDRESS_WIDTH = 34,
    parameter AXI_DATA_WIDTH    = 512,
    parameter MAX_DIMENSION = 1023
) (
    input logic core_clk,
    input logic resetn,

    // Request interface
    input  logic matrix_bank_req_valid,
    output logic matrix_bank_req_ready,
    input  REQ_t matrix_bank_req,

    // Response interface
    output logic  matrix_bank_resp_valid,
    output RESP_t matrix_bank_resp,

    // AXI Read Master Request interface
    output logic                                        axi_rm_fetch_req_valid,
    input  logic                                        axi_rm_fetch_req_ready,
    output logic [               AXI_ADDRESS_WIDTH-1:0] axi_rm_fetch_start_address,
    output logic [$clog2(MAX_FETCH_REQ_BYTE_COUNT)-1:0] axi_rm_fetch_byte_count,

    // AXI Read Master Response interface
    input  logic                      axi_rm_fetch_resp_valid,
    output logic                      axi_rm_fetch_resp_ready,
    input  logic                      axi_rm_fetch_resp_last,
    input  logic [AXI_DATA_WIDTH-1:0] axi_rm_fetch_resp_data,
    input  logic [               3:0] axi_rm_fetch_resp_axi_id,

    // Row channel requests
    input  logic             row_channel_req_valid,
    output logic             row_channel_req_ready,
    input  ROW_CHANNEL_REQ_t row_channel_req,

    // Row channel responses
    output logic              row_channel_resp_valid,
    input  logic              row_channel_resp_ready,
    output ROW_CHANNEL_RESP_t row_channel_resp

);

  typedef enum logic [3:0] {
    MATRIX_BANK_FSM_IDLE           = 4'd0,
    MATRIX_BANK_FSM_FETCH_REQ      = 4'd1,
    MATRIX_BANK_FSM_WAIT_RESP      = 4'd2,
    MATRIX_BANK_FSM_WRITE          = 4'd3,
    MATRIX_BANK_FSM_MATRIX_WAITING = 4'd4,
    MATRIX_BANK_FSM_DUMP_ROWS      = 4'd5
  } MATRIX_BANK_FSM_e;

  // ==================================================================================================================================================
  // Declarations
  // ==================================================================================================================================================

  REQ_t matrix_bank_req_q;
  MATRIX_BANK_FSM_e matrix_bank_state, matrix_bank_state_n;
  logic                                                          accepting_row_channel_resp;

  // Matrix row FIFOs
  // ----------------------------------------------------------------

  logic [            MAX_DIMENSION-1:0]                          row_fifo_push;
  logic [                         31:0]                          row_fifo_in_data;

  logic [            MAX_DIMENSION-1:0]                          row_fifo_pop;
  logic [            MAX_DIMENSION-1:0]                          row_fifo_out_valid;
  logic [            MAX_DIMENSION-1:0][                   31:0] row_fifo_out_data;

  logic [            MAX_DIMENSION-1:0][$clog2(MAX_DIMENSION):0] row_fifo_count;
  logic [            MAX_DIMENSION-1:0]                          row_fifo_empty;
  logic [            MAX_DIMENSION-1:0]                          row_fifo_full;

  // Fetching
  // ----------------------------------------------------------------

  logic [      $clog2(MAX_DIMENSION):0]                          features_written;
  logic [      $clog2(MAX_DIMENSION):0]                          rows_fetched;
  logic [$clog2(AXI_DATA_WIDTH/32)-1:0]                          feature_offset;
  logic [                          4:0]                          expected_responses;
  logic                                                          axi_rm_fetch_resp_last_q;
  logic [           AXI_DATA_WIDTH-1:0]                          axi_rm_fetch_resp_data_q;
  logic [                          3:0]                          axi_rm_fetch_resp_axi_id_q;


  logic [  $clog2(MAX_DIMENSION*4)-1:0]                          bytes_per_row;
  logic [  $clog2(MAX_DIMENSION*4)-1:0]                          bytes_per_row_padded;

  // Driving row channel responses
  logic [    $clog2(MAX_DIMENSION)-1:0]                          required_pulses;
  logic [            MAX_DIMENSION-1:0]                          row_pop_shift;
  logic [      $clog2(MAX_DIMENSION):0]                          row_counter;

  logic                                                          reset_matrix;

  // ==================================================================================================================================================
  // Instances
  // ==================================================================================================================================================

  for (genvar row = 0; row < MAX_DIMENSION; row++) begin
    ultraram_fifo #(
        .WIDTH(32),
        .DEPTH(MAX_DIMENSION)
    ) matrix_row (
        .core_clk,
        .resetn,

        .push   (row_fifo_push[row]),
        .in_data(row_fifo_in_data),

        .pop           (row_fifo_pop[row]),
        .reset_read_ptr(reset_matrix),
        .out_valid     (row_fifo_out_valid[row]),
        .out_data      (row_fifo_out_data[row]),

        .count(row_fifo_count[row]),
        .empty(row_fifo_empty[row]),
        .full (row_fifo_full[row])
    );

    assign row_fifo_push      [row] = (matrix_bank_state == MATRIX_BANK_FSM_WRITE) & (row == (rows_fetched - 1));

    // Pop FIFO when currently valid rows are being dumped
    assign row_fifo_pop       [row] = (matrix_bank_state == MATRIX_BANK_FSM_DUMP_ROWS) && row_pop_shift[row] && !row_fifo_empty[row] && accepting_row_channel_resp;
  end

  // ==================================================================================================================================================
  // Logic
  // ==================================================================================================================================================

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      matrix_bank_state <= MATRIX_BANK_FSM_IDLE;
    end else begin
      matrix_bank_state <= matrix_bank_state_n;
    end
  end

  always_comb begin
    matrix_bank_state_n = matrix_bank_state;

    case (matrix_bank_state)

      MATRIX_BANK_FSM_IDLE: begin
        matrix_bank_state_n = matrix_bank_req_valid ? MATRIX_BANK_FSM_FETCH_REQ : MATRIX_BANK_FSM_IDLE;
      end

      MATRIX_BANK_FSM_FETCH_REQ: begin
        matrix_bank_state_n = axi_rm_fetch_req_ready ? MATRIX_BANK_FSM_WAIT_RESP : MATRIX_BANK_FSM_FETCH_REQ;
      end

      MATRIX_BANK_FSM_WAIT_RESP: begin
        matrix_bank_state_n = axi_rm_fetch_resp_valid ? MATRIX_BANK_FSM_WRITE : MATRIX_BANK_FSM_WAIT_RESP;
      end

      MATRIX_BANK_FSM_WRITE: begin
        matrix_bank_state_n =
        // Finished writing all features for entire matrix
        (feature_offset == 4'd15) && (rows_fetched == matrix_bank_req_q.rows) && (expected_responses == '0) ? MATRIX_BANK_FSM_MATRIX_WAITING

        // Finished writing all features for current row
        : (feature_offset == 4'd15) && (expected_responses == '0) ? MATRIX_BANK_FSM_FETCH_REQ

        // Finished writing all features for current AXI response beat
        : (feature_offset == 4'd15) ? MATRIX_BANK_FSM_WAIT_RESP

        // Still writing features for current AXI response beat
        : MATRIX_BANK_FSM_WRITE;
      end

      MATRIX_BANK_FSM_MATRIX_WAITING: begin
        matrix_bank_state_n = row_channel_req_valid ? MATRIX_BANK_FSM_DUMP_ROWS : MATRIX_BANK_FSM_MATRIX_WAITING;
      end

      MATRIX_BANK_FSM_DUMP_ROWS: begin
        matrix_bank_state_n = row_channel_resp_valid && row_channel_resp_ready && row_channel_resp.done ? MATRIX_BANK_FSM_MATRIX_WAITING
                            : MATRIX_BANK_FSM_DUMP_ROWS;
      end

    endcase
  end

  // Requests interface
  // ----------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      matrix_bank_req_q <= '0;

      // Receiving a request
    end else if (matrix_bank_req_valid && matrix_bank_req_ready) begin
      matrix_bank_req_q <= matrix_bank_req;
    end
  end

  always_comb begin
    matrix_bank_req_ready = (matrix_bank_state == MATRIX_BANK_FSM_IDLE);

    matrix_bank_resp_valid = (matrix_bank_state == MATRIX_BANK_FSM_WRITE) && (matrix_bank_state_n == MATRIX_BANK_FSM_MATRIX_WAITING);
    matrix_bank_resp.partial = ~row_fifo_full;  // 0 when row FIFOs are saturated

  end

  // Read master fetch request logic
  // ----------------------------------------------------------------

  always_comb begin
    axi_rm_fetch_req_valid = (matrix_bank_state == MATRIX_BANK_FSM_FETCH_REQ);
    axi_rm_fetch_byte_count = matrix_bank_req_q.columns * 4;

    bytes_per_row = matrix_bank_req_q.columns * 4;
    bytes_per_row_padded = {bytes_per_row[$clog2(MAX_DIMENSION*4)-1:6], 6'b0} +
        (|bytes_per_row[5:0] ? 'd64 : '0);  // round up to nearest multiple of 64
    axi_rm_fetch_start_address = matrix_bank_req_q.start_address + rows_fetched * bytes_per_row_padded;
  end

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      rows_fetched <= '0;
      expected_responses <= '0;

      axi_rm_fetch_resp_last_q <= '0;
      axi_rm_fetch_resp_data_q <= '0;
      axi_rm_fetch_resp_axi_id_q <= '0;

      // Accepting request to axi read master
    end else if (axi_rm_fetch_req_valid && axi_rm_fetch_req_ready) begin
      rows_fetched <= rows_fetched + 1'b1;

      // divide by 16 (features per AXI beat), round up
      expected_responses <= matrix_bank_req_q.columns[$clog2(
          MAX_DIMENSION
      ):4] + (|matrix_bank_req_q.columns[3:0] ? 1'b1 : 1'b0);

      // Accepting response from axi read master
    end else if (axi_rm_fetch_resp_valid && axi_rm_fetch_resp_ready) begin
      expected_responses <= expected_responses - 1'b1;

      // Register response payloads
      axi_rm_fetch_resp_last_q <= axi_rm_fetch_resp_last;
      axi_rm_fetch_resp_data_q <= axi_rm_fetch_resp_data;
      axi_rm_fetch_resp_axi_id_q <= axi_rm_fetch_resp_axi_id;
    end
  end

  assign axi_rm_fetch_resp_ready = (matrix_bank_state == MATRIX_BANK_FSM_WAIT_RESP);

  // FIFO write logic
  // ----------------------------------------------------------------

  always_comb begin
    row_fifo_in_data = 
                        feature_offset == '0        ? axi_rm_fetch_resp_data_q [511 : 480]
                        : feature_offset == 1'b1    ? axi_rm_fetch_resp_data_q [479 : 448]
                        : feature_offset == 2'b10   ? axi_rm_fetch_resp_data_q [447 : 416]
                        : feature_offset == 2'b11   ? axi_rm_fetch_resp_data_q [415 : 384]
                        : feature_offset == 3'b100  ? axi_rm_fetch_resp_data_q [383 : 352]
                        : feature_offset == 3'b101  ? axi_rm_fetch_resp_data_q [351 : 320]
                        : feature_offset == 3'b110  ? axi_rm_fetch_resp_data_q [319 : 288]
                        : feature_offset == 3'b111  ? axi_rm_fetch_resp_data_q [287 : 256]
                        : feature_offset == 4'b1000 ? axi_rm_fetch_resp_data_q [255 : 224]
                        : feature_offset == 4'b1001 ? axi_rm_fetch_resp_data_q [223 : 192]
                        : feature_offset == 4'b1010 ? axi_rm_fetch_resp_data_q [191 : 160]
                        : feature_offset == 4'b1011 ? axi_rm_fetch_resp_data_q [159 : 128]
                        : feature_offset == 4'b1100 ? axi_rm_fetch_resp_data_q [127 : 96]
                        : feature_offset == 4'b1101 ? axi_rm_fetch_resp_data_q [95 : 64]
                        : feature_offset == 4'b1110 ? axi_rm_fetch_resp_data_q [63 : 32]
                        : feature_offset == 4'b1111 ? axi_rm_fetch_resp_data_q [31 : 0]
                        : '0;
  end

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      feature_offset   <= '0;
      features_written <= '0;

      // Write into FIFO
    end else if (matrix_bank_state == MATRIX_BANK_FSM_WRITE) begin
      feature_offset   <= feature_offset == 4'd15 ? '0 : (feature_offset + 1'b1);
      features_written <= features_written + 1'b1;
    end
  end

  // Row dumping through Row Channel
  // ----------------------------------------------------------------

  assign row_channel_req_ready = (matrix_bank_state == MATRIX_BANK_FSM_MATRIX_WAITING);

  // Shift register to flush through matrix diagonally
  for (genvar row = 1; row < MAX_DIMENSION; row++) begin
    always_ff @(posedge core_clk or negedge resetn) begin
      if (!resetn) begin
        row_pop_shift[row] <= '0;

        // Clear shift register when starting new row dump
      end else if ((matrix_bank_state == MATRIX_BANK_FSM_MATRIX_WAITING) && (matrix_bank_state_n == MATRIX_BANK_FSM_DUMP_ROWS)) begin
        row_pop_shift[row] <= '0;

        // Shift register when accepting row channel response
      end else if (row_channel_resp_valid && row_channel_resp_ready) begin
        row_pop_shift[row] <= row_pop_shift[row-1];
      end
    end
  end

  // Round up in features to the nearest multiple of 16
  assign required_pulses = {matrix_bank_req_q.columns[$clog2(
      MAX_DIMENSION
  )-1:4], 4'd0} + (|matrix_bank_req_q.columns[3:0] ? 'd16 : '0);

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      row_pop_shift[0] <= '0;  // Head of shift register
      row_counter <= '0;

      // Starting new feature dump, reset all flags and counters
    end else if ((matrix_bank_state == MATRIX_BANK_FSM_MATRIX_WAITING) && (matrix_bank_state_n == MATRIX_BANK_FSM_DUMP_ROWS)) begin
      row_pop_shift[0] <= '1;
      row_counter <= '0;

      // Pulse matrix
    end else if (|row_fifo_pop) begin
      // Increment when popping any rows, but latch at '1
      row_counter <= (row_counter == required_pulses) ? row_counter : (row_counter + 1'b1);

      // If accepting row channel response, new data is available on all row FIFOs
      // so shift register and clear popped rows flag
      row_pop_shift[0] <= !(row_counter >= (required_pulses - 1'b1));

    end
  end

  always_comb begin
    // Issue row channel response when new data is available on all row FIFOs following a pop
    row_channel_resp_valid = (matrix_bank_state == MATRIX_BANK_FSM_DUMP_ROWS) && (&row_fifo_out_valid) && |row_pop_shift;

    row_channel_resp.data = row_fifo_out_data;
    row_channel_resp.valid_mask = row_pop_shift & ~row_fifo_empty;

    row_channel_resp.done = (row_channel_resp.valid_mask == '0);

    accepting_row_channel_resp = (row_channel_resp_valid && row_channel_resp_ready);
  end

  // When finished dumping matrix, reset read pointer so the same matrix can be used for the next FTE pass
  assign reset_matrix = (matrix_bank_state == MATRIX_BANK_FSM_DUMP_ROWS) && (matrix_bank_state_n == MATRIX_BANK_FSM_MATRIX_WAITING);

endmodule
