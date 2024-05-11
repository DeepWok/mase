/*
Module      : fifo
Description : This module implements a max throughput streaming fifo with
              registered output.

              Assumes that the simple_dual_port_ram backing it has a one cycle
              read latency.
*/

`timescale 1ns / 1ps

module fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 16,
    parameter SIZE       = DEPTH
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready,

    output logic empty,
    output logic full
);

  localparam ADDR_WIDTH = SIZE == 1 ? 1 : $clog2(SIZE);
  localparam PTR_WIDTH = ADDR_WIDTH + 1;

  typedef struct packed {
    logic [DATA_WIDTH-1:0] data;
    logic valid;
  } reg_t;

  typedef struct packed {
    // Write state
    logic [PTR_WIDTH-1:0] write_ptr;
    logic [ADDR_WIDTH:0]  size;

    // Read state
    logic [PTR_WIDTH-1:0] read_ptr;
    logic ram_dout_valid;  // Pulse signal for ram reads

    // Controls the next register to be connected to output
    logic next_reg;

    // Output register
    reg_t out_reg;

    // Extra register required to buffer the output of RAM due to delay
    reg_t extra_reg;
  } self_t;

  self_t self, next_self;

  // Ram signals
  logic [DATA_WIDTH-1:0] ram_rd_dout;
  logic pause_reads;

  always_comb begin
    next_self = self;

    // Input side ready
    in_ready = self.size != SIZE - 1;

    // Pause reading when there is (no transfer on this cycle) AND the registers are full.
    pause_reads = !out_ready && (self.out_reg.valid || self.extra_reg.valid);

    next_self.write_ptr = in_valid && in_ready && (self.write_ptr == SIZE - 1) ? '0
                          : in_valid && in_ready ? next_self.write_ptr + 1'b1
                          : self.write_ptr;

    next_self.size = (|self.size && !pause_reads) ? self.size - 1'b1
                    : (in_valid && in_ready) ? self.size + 1'b1
                    : self.size;

    next_self.read_ptr = (|self.size && !pause_reads) && (self.read_ptr == SIZE - 1) ? '0
                      : (|self.size && !pause_reads) ? self.read_ptr + 1'b1
                      : self.read_ptr;

    next_self.ram_dout_valid = (|self.size && !pause_reads);

    // Output register
    next_self.out_reg.data = !(self.out_reg.valid && !out_ready) ? ram_rd_dout : self.out_reg.data;

    next_self.out_reg.valid = !self.next_reg && (out_ready && self.out_reg.valid) ? self.ram_dout_valid
                              : self.ram_dout_valid && !(self.out_reg.valid && !out_ready) ? '1
                              : self.out_reg.valid;

    // Extra register
    next_self.extra_reg.data = self.ram_dout_valid && self.out_reg.valid && !out_ready ? ram_rd_dout
                              : self.extra_reg.data;

    next_self.extra_reg.valid = self.next_reg && out_ready && self.extra_reg.valid ? '0
                              : self.ram_dout_valid && self.out_reg.valid && !out_ready ? '1
                              : self.extra_reg.valid;

    // Output interface
    out_data = self.next_reg ? self.extra_reg.data : self.out_reg.data;
    out_valid = self.next_reg ? self.extra_reg.valid : self.out_reg.valid;

    // Toggle between out/extra reg
    next_self.next_reg = !self.next_reg && out_ready && self.out_reg.valid && self.extra_reg.valid ? '1
                        : self.next_reg && out_ready && self.extra_reg.valid ? '0
                        : self.next_reg;
  end

  simple_dual_port_ram #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH),
      .SIZE      (SIZE)
  ) ram_inst (
      .clk    (clk),
      .wr_addr(self.write_ptr),
      .wr_din (in_data),
      .wr_en  (in_valid && in_ready),
      .rd_addr(self.read_ptr),
      .rd_dout(ram_rd_dout)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '0;
    end else begin
      self <= next_self;
    end
  end

  assign empty = (self.size == 0);
  assign full  = (self.size == SIZE);
endmodule
