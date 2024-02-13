/*
Module      : repeat_circular_buffer
Description : This module is a repeating circular buffer.
*/

`timescale 1ns/1ps

module repeat_circular_buffer #(
    parameter DATA_WIDTH = 32,
    parameter REPEAT = 2,
    parameter SIZE = 4
) (
    input logic clk,
    input logic rst,

    // Input streaming port
    input  logic [DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    // Output streaming port
    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);

localparam REPS_WIDTH = $clog2(REPEAT);
localparam ADDR_WIDTH = SIZE == 1 ? 1 : $clog2(SIZE);

// Coded for RAM read latency of 1 cycle
localparam READ_LATENCY = 1;

// Top bit signifies write done
localparam WRITE_PTR_WIDTH = ADDR_WIDTH + 1;

initial begin
    assert (REPEAT > 1);
    assert (READ_LATENCY == 1) else $fatal("Currently not supported.");
end

typedef struct {
    logic [DATA_WIDTH-1:0] data;
    logic valid;
} extra_reg_t;

struct {
    // Write state
    logic [WRITE_PTR_WIDTH-1:0] write_ptr;
    logic [ADDR_WIDTH:0] size;

    // Read state
    logic [ADDR_WIDTH-1:0] read_ptr;
    logic [REPS_WIDTH-1:0] rep;
    logic rd_valid;

    // Extra register required to buffer the output of RAM due to delay
    extra_reg_t extra_reg;
} self, next_self;

// Ram signals
logic ram_wr_en;
logic [DATA_WIDTH-1:0] ram_rd_dout;

// Backpressure control signal
logic pause_reads;

// Output register slice
logic [DATA_WIDTH-1:0] reg_in_data, reg_out_data;
logic reg_in_valid, reg_out_valid;
logic reg_in_ready, reg_out_ready;

always_comb begin
    next_self = self;

    // Input side ready
    in_ready = self.rep == 0 && self.size != SIZE;

    // Write side of machine
    // Increment write pointer
    if (in_valid && in_ready) begin
        next_self.write_ptr = self.write_ptr + 1;
        next_self.size = self.size + 1;
        ram_wr_en = 1;
    end else begin
        ram_wr_en = 0;
    end

    // Read side of machine
    if (self.read_ptr < self.write_ptr && !pause_reads) begin
        if (self.read_ptr == SIZE-1 && self.rep == REPEAT-1) begin
            // FULL RESET
            next_self = '{default: 0};
        end else if (self.read_ptr == SIZE-1) begin
            next_self.read_ptr = 0;
            next_self.rep += 1;
        end else begin
            next_self.read_ptr += 1;
        end
        next_self.rd_valid = 1;
    end else begin
        next_self.rd_valid = 0;
    end

    // We need to store this extra value that spills out due to the read latency
    // being 1. If the latency was N, we would need N spill over registers.
    if (pause_reads && self.rd_valid) begin
        next_self.extra_reg.data = ram_rd_dout;
        next_self.extra_reg.valid = 1;
    end

    // Clearing value in extra spill over register
    if (self.extra_reg.valid) begin
        reg_in_data = self.extra_reg.data;
        reg_in_valid = 1;
        if (reg_in_ready) begin
            // Reset back if transfer happened: reg_in_valid && reg_in_ready
            next_self.extra_reg.valid = 0;
        end
    end else begin
        reg_in_data = ram_rd_dout;
        reg_in_valid = self.rd_valid;
    end

end

register_slice #(
    .DATA_WIDTH (DATA_WIDTH)
) reg_slice_inst (
    .clk        (clk),
    .rst        (rst),
    .in_data    (reg_in_data),
    .in_valid   (reg_in_valid),
    .in_ready   (reg_in_ready),
    .out_data   (reg_out_data),
    .out_valid  (reg_out_valid),
    .out_ready  (reg_out_ready)
);

// Pause reading when there is (no transfer on this cycle AND we already have
// valid data in the output buffer) OR the extra spill over register is full.
assign pause_reads = (!out_ready && reg_out_valid) || self.extra_reg.valid;
assign out_data = reg_out_data;
assign out_valid = reg_out_valid;
assign reg_out_ready = out_ready;

simple_dual_port_ram #(
    .DATA_WIDTH   (DATA_WIDTH),
    .ADDR_WIDTH   (ADDR_WIDTH),
    .SIZE         (SIZE)
) ram_inst (
    .clk        (clk),
    .wr_addr    (self.write_ptr),
    .wr_din     (in_data),
    .wr_en      (ram_wr_en),
    .rd_addr    (self.read_ptr),
    .rd_dout    (ram_rd_dout)
);

always_ff @(posedge clk) begin
    if (rst) begin
        self <= '{default: 0};
    end else begin
        self <= next_self;
    end
end

endmodule
