/*
Module      : repeat_circular_buffer
Description : This module is a repeating circular buffer.
*/

`timescale 1ns / 1ps

module repeat_circular_buffer_programmable #(
    parameter DATA_WIDTH = 32,
    parameter MAX_REPEAT = 2,
    parameter MAX_SIZE = 4
) (
    input logic clk,
    input logic rst,

    //Programmable parameters
    input logic [REPS_WIDTH:0] repeat_n,
    input logic [SIZE_WIDTH:0] size_n,

    // Input streaming port
    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    // Output streaming port
    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);


  localparam SIZE_WIDTH = $clog2(MAX_SIZE);
  localparam REPS_WIDTH = $clog2(MAX_REPEAT);
  localparam ADDR_WIDTH = MAX_SIZE == 1 ? 1 : $clog2(MAX_SIZE);
  localparam PTR_WIDTH = ADDR_WIDTH + 1;

  typedef struct {
    logic [DATA_WIDTH-1:0] data;
    logic valid;
  } reg_t;

  struct {
    // Write state
    logic [PTR_WIDTH-1:0] write_ptr;
    logic [ADDR_WIDTH:0]  size;

    // Read state
    logic [PTR_WIDTH-1:0] read_ptr;
    logic [REPS_WIDTH-1:0] rep;
    logic ram_dout_valid;  // Pulse signal for ram reads

    // Controls the next register to be connected to output
    logic next_reg;

    // Output register
    reg_t out_reg;

    // Extra register required to buffer the output of RAM due to delay
    reg_t extra_reg;
  }
      self, next_self;

  // Ram signals
  logic ram_wr_en;
  logic [DATA_WIDTH-1:0] ram_rd_dout;

  // Backpressure control signal
  logic pause_reads;

  always_comb begin
    next_self = self;

    // Input side ready
    if(repeat_n==1)
      in_ready = (self.size != SIZE);
    else
      in_ready = (self.size != size_n) && !(self.rep == repeat_n - 1 && self.write_ptr == self.read_ptr);

    // Pause reading when there is (no transfer on this cycle) AND the registers are full.
    pause_reads = !out_ready && (self.out_reg.valid || self.extra_reg.valid);

    // Write side of machine
    if (self.write_ptr == size_n) begin
      // Write is finished, reset when we are on the last rep
      if (self.rep == repeat_n - 1) begin
        next_self.write_ptr = 0;
      end
      ram_wr_en = 0;
    end else begin
      // Still need to write into ram
      if (in_valid && in_ready) begin
        next_self.write_ptr = self.write_ptr + 1;
        next_self.size = self.size + 1;
        ram_wr_en = 1;
      end else begin
        ram_wr_en = 0;
      end
    end

    // Read side of machine
    // Conditions:
    // - There is data in the RAM
    // - AND there is no backpressure from output regs (pause_reads)
    // - AND we are not ahead of write ptr on the first rep where write side of
    //   machine is still writing into RAM. This is to prevent read and write to
    //   the same address at the same time, yielding invalid data.
    if (self.size != 0 && !pause_reads && !(self.rep == 0 && self.read_ptr == self.write_ptr)) begin
      if (self.read_ptr == size_n - 1 && self.rep == repeat_n - 1) begin
        next_self.read_ptr = 0;
        next_self.rep = 0;
      end else if (self.read_ptr == size_n - 1) begin
        next_self.read_ptr = 0;
        next_self.rep += 1;
      end else begin
        next_self.read_ptr += 1;
      end
      // Decrease fifo size only on last rep
      if (self.rep == repeat_n - 1) begin
        next_self.size -= 1;
      end
      next_self.ram_dout_valid = 1;
    end else begin
      next_self.ram_dout_valid = 0;
    end
    // end

    // Input mux for extra reg
    if (self.ram_dout_valid) begin
      if (self.out_reg.valid && !out_ready) begin
        next_self.extra_reg.data  = ram_rd_dout;
        next_self.extra_reg.valid = 1;
      end else begin
        next_self.out_reg.data  = ram_rd_dout;
        next_self.out_reg.valid = 1;
      end
    end

    // Output mux for extra reg
    if (self.next_reg) begin
      out_data  = self.extra_reg.data;
      out_valid = self.extra_reg.valid;
      if (out_ready && self.extra_reg.valid) begin
        next_self.extra_reg.valid = 0;
        next_self.next_reg = 0;
      end
    end else begin
      out_data  = self.out_reg.data;
      out_valid = self.out_reg.valid;
      if (out_ready && self.out_reg.valid) begin
        next_self.out_reg.valid = self.ram_dout_valid;
        if (self.extra_reg.valid) begin
          next_self.next_reg = 1;
        end
      end
    end

  end

  simple_dual_port_ram #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH),
      .SIZE      (MAX_SIZE)
  ) ram_inst (
      .clk    (clk),
      .wr_addr(self.write_ptr),
      .wr_din (in_data),
      .wr_en  (ram_wr_en),
      .rd_addr(self.read_ptr),
      .rd_dout(ram_rd_dout)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{default: 0};
    end else begin
      self <= next_self;
    end
  end

endmodule
