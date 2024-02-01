
module axi_write_master (
    input logic core_clk,
    input logic resetn,

    input  logic        axi_write_master_req_valid,
    output logic        axi_write_master_req_ready,
    input  logic [33:0] axi_write_master_req_start_address,
    input  logic [ 7:0] axi_write_master_req_len,

    output logic         data_queue_pop,
    input  logic         data_queue_data_valid,
    input  logic [511:0] data_queue_data,

    output logic axi_write_master_resp_valid,
    input  logic axi_write_master_resp_ready,

    // AXI Write Master -> AXI Interconnect
    output logic [33:0] axi_awaddr,
    output logic [ 1:0] axi_awburst,
    output logic [ 3:0] axi_awcache,
    output logic [ 3:0] axi_awid,
    output logic [ 7:0] axi_awlen,
    output logic [ 0:0] axi_awlock,
    output logic [ 2:0] axi_awprot,
    output logic [ 3:0] axi_awqos,
    input  logic        axi_awready,
    output logic [ 2:0] axi_awsize,
    output logic        axi_awvalid,

    output logic [511:0] axi_wdata,
    output logic         axi_wlast,
    input  logic         axi_wready,
    output logic [ 63:0] axi_wstrb,
    output logic         axi_wvalid,

    input  logic [3:0] axi_bid,
    output logic       axi_bready,
    input  logic [1:0] axi_bresp,
    input  logic       axi_bvalid,

    output logic [33:0] axi_araddr,
    output logic [ 1:0] axi_arburst,
    output logic [ 3:0] axi_arcache,
    output logic [ 3:0] axi_arid,
    output logic [ 7:0] axi_arlen,
    output logic [ 0:0] axi_arlock,
    output logic [ 2:0] axi_arprot,
    output logic [ 3:0] axi_arqos,
    output logic [ 2:0] axi_arsize,
    output logic        axi_arvalid,
    input  logic        axi_arready,

    input  logic [511:0] axi_rdata,
    input  logic [  3:0] axi_rid,
    input  logic         axi_rlast,
    output logic         axi_rready,
    input  logic [  1:0] axi_rresp,
    input  logic         axi_rvalid
);

  typedef enum logic [1:0] {
    AXI_IDLE,
    AXI_AW,
    AXI_W,
    AXI_B
  } AXI_WRITE_STATE_e;

  // ==================================================================================================================================================
  // Declarations
  // ==================================================================================================================================================

  AXI_WRITE_STATE_e axi_write_state, axi_write_state_n;

  logic [33:0] axi_write_master_req_start_address_q;
  logic [ 7:0] axi_write_master_req_len_q;
  logic [ 7:0] sent_beats;

  // ==================================================================================================================================================
  // Logic
  // ==================================================================================================================================================

  // Register request payloads
  // ---------------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      axi_write_master_req_start_address_q <= '0;
      axi_write_master_req_len_q           <= '0;

      // Accepting write request
    end else if (axi_write_master_req_valid && axi_write_master_req_ready) begin
      axi_write_master_req_start_address_q <= axi_write_master_req_start_address;
      axi_write_master_req_len_q           <= axi_write_master_req_len;
    end
  end

  // State Machine
  // ---------------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      axi_write_state <= AXI_IDLE;
    end else begin
      axi_write_state <= axi_write_state_n;
    end
  end

  always_comb begin
    axi_write_state_n = axi_write_state;

    case (axi_write_state)
      AXI_IDLE: begin
        axi_write_state_n = axi_write_master_req_valid && axi_write_master_req_ready ? AXI_AW
                              : AXI_IDLE;
      end

      AXI_AW: begin
        axi_write_state_n = axi_awready ? AXI_W : AXI_AW;
      end

      AXI_W: begin
        axi_write_state_n = axi_wvalid && axi_wready && axi_wlast ? AXI_B : AXI_W;
      end

      AXI_B: begin
        axi_write_state_n = axi_bvalid ? AXI_IDLE : AXI_B;
      end

    endcase

  end

  assign axi_write_master_req_ready = (axi_write_state == AXI_IDLE);

  // Drive AXI signals
  // ---------------------------------------------------------------------

  always_comb begin
    axi_awvalid = (axi_write_state == AXI_AW);

    axi_awaddr = axi_write_master_req_start_address_q;

    axi_awsize  = 3'b110; // 64 bytes
    axi_awburst = 2'b01; // INCR mode (increment by 64 bytes for each beat)
    axi_awlen   = axi_write_master_req_len_q;

    // Unused features
    axi_awcache = '0;
    axi_awid    = '0;
    axi_awlock  = '0;
    axi_awprot  = '0;
    axi_awqos   = '0;

    axi_wvalid = (axi_write_state == AXI_W) && data_queue_data_valid;
    axi_wdata  = data_queue_data;
    axi_wlast  = (sent_beats == axi_write_master_req_len_q);
    axi_wstrb  = '1;

    axi_bready = (axi_write_state == AXI_B);
  end

  // Queue interface
  // ---------------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      sent_beats <= '0;

      // Starting to flush row from systolic array
    end else if (axi_write_master_req_valid && axi_write_master_req_ready) begin
      sent_beats <= '0;

      // Finished current systolic module
    end else if (axi_wvalid && axi_wready) begin
      sent_beats <= sent_beats + 1'b1;
    end
  end

  always_comb begin
    data_queue_pop = (axi_write_state == AXI_W) && axi_wvalid && axi_wready;
    axi_write_master_resp_valid = (axi_write_state == AXI_B) && axi_bvalid;
  end

  // Write-Only Interface
  // ---------------------------------------------------------------------

  always_comb begin
    axi_araddr  = '0;
    axi_arburst = '0;
    axi_arcache = '0;
    axi_arid    = '0;
    axi_arlen   = '0;
    axi_arlock  = '0;
    axi_arprot  = '0;
    axi_arqos   = '0;
    axi_arsize  = '0;
    axi_arvalid = '0;
    axi_rready  = '0;
  end

endmodule
