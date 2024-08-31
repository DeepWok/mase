`timescale 1ns / 1ps
module fixed_roll_buffer #(
    parameter ROLL_MAX_DISTANCE = 64,
    parameter MAX_BUFFER_SIZE = 256,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,   
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,   
    parameter DATA_IN_0_PRECISION_0 = 16,  
    parameter DATA_IN_0_PRECISION_1 = 8,

    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 2,
    parameter COUNTER_SIZE = 4,

    localparam ADDR_RANGE = $clog2(MAX_BUFFER_SIZE),
    localparam ADDR_WIDTH = $clog2(ADDR_RANGE)
)
(

    input clk,
    input rst,

    input logic [$clog2(ROLL_MAX_DISTANCE):0] roll_distance,
    input logic [$clog2(MAX_BUFFER_SIZE):0]   buffer_size,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid, 
    output logic data_in_0_ready,
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready,

    output logic done
);

 typedef struct packed {
    logic [DATA_IN_0_PRECISION_0*DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1:0] data;
    logic valid;
  } reg_t;

  typedef struct packed {
    // Write state
    logic [ADDR_RANGE:0] write_ptr;
    logic [$clog2(MAX_BUFFER_SIZE):0]  size;

    // Read state
    logic [ADDR_RANGE:0] read_ptr;
    logic ram_dout_valid;  // Pulse signal for ram reads

    // Can't use enum becuase of reset syntax
    logic [1:0]state;

    logic [COUNTER_SIZE:0] write_counter;
    logic [$clog2(MAX_BUFFER_SIZE):0] prev_buffer_size;

    // Controls the next register to be connected to output
    logic next_reg;

    // Output register
    reg_t out_reg;

    // Extra register required to buffer the output of RAM due to delay
    reg_t extra_reg;
  } self_t;

  self_t self, next_self;

  // Ram signals
  logic ram_wr_en;
  logic [DATA_IN_0_PRECISION_0-1:0] ram_rd_dout [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1];
  logic [DATA_IN_0_PRECISION_0*DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0] ram_rd_dout_flattened;
  logic [DATA_IN_0_PRECISION_0*DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0] data_out_flattened;
  logic [DATA_IN_0_PRECISION_0*DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0] data_in_flattened;
  logic empty;

  // Backpressure control signal
  logic pause_reads;

  matrix_flatten #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .DIM0      (DATA_IN_0_PARALLELISM_DIM_0),
      .DIM1      (DATA_IN_0_PARALLELISM_DIM_1)
  ) input_flatten (
      .data_in (data_in_0),
      .data_out(data_in_flattened)
  );

matrix_unflatten #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .DIM0      (DATA_IN_0_PARALLELISM_DIM_0),
      .DIM1      (DATA_IN_0_PARALLELISM_DIM_1)
  ) output_flatten (
    .data_in (data_out_flattened),
    .data_out (data_out_0)
);



  always_comb begin
    next_self = self;

    if(self.state == 1)
      next_self.write_counter <= 0;
    if(data_in_0_ready && data_in_0_valid)
      next_self.write_counter <= next_self.write_counter + 1;  

    if(self.state == 1) begin
      next_self.prev_buffer_size = buffer_size;
      next_self.read_ptr = 0;
      done = 1;
    end else
      done = 0;

    if(self.state == 2)begin
      next_self.write_ptr = roll_distance;
    end

    // Input side ready
    data_in_0_ready = (self.write_counter <= buffer_size) && (self.state == 0);

    // Pause reading when there is (no transfer on this cycle) AND the registers are full.
    pause_reads = (!data_out_0_ready && (self.out_reg.valid || self.extra_reg.valid)) || self.read_ptr >= self.prev_buffer_size;

    if(empty) next_self.write_ptr = roll_distance;

    // Write side of machine
    // Increment write pointer
    if (data_in_0_valid && data_in_0_ready && self.state == 0) begin
      if (self.write_ptr == buffer_size - 1) begin
        next_self.write_ptr = 0;
      end else begin
        next_self.write_ptr += 1;
      end
      next_self.size = self.size + 1;
      ram_wr_en = 1;
    end else begin
      ram_wr_en = 0;
    end

    // Read side of machine
    if (!pause_reads && self.state == 0) begin
      next_self.read_ptr += 1;
      next_self.ram_dout_valid = 1;
    end else begin
      next_self.ram_dout_valid = 0;
    end

    // Input mux for extra reg
    if (self.ram_dout_valid) begin
      if (self.out_reg.valid && !data_out_0_ready) begin
        next_self.extra_reg.data  = ram_rd_dout_flattened;
        next_self.extra_reg.valid = 1;
      end else begin
        next_self.out_reg.data  = ram_rd_dout_flattened;
        next_self.out_reg.valid = 1;
      end
    end

    // Output mux for extra reg
    if (self.next_reg) begin
      data_out_flattened  = self.extra_reg.data;
      data_out_0_valid = self.extra_reg.valid;
      if (data_out_0_ready && self.extra_reg.valid) begin
        next_self.extra_reg.valid = 0;
        next_self.next_reg = 0;
      end
    end else begin
      data_out_flattened  = self.out_reg.data;
      data_out_0_valid = self.out_reg.valid;
      if (data_out_0_ready && self.out_reg.valid) begin
        next_self.out_reg.valid = self.ram_dout_valid;
        if (self.extra_reg.valid) begin
          next_self.next_reg = 1;
        end
      end
    end

    case(self.state) 
      0: if(self.write_counter == buffer_size-1)  next_self.state = 1;
                else next_self.state = 0;
      1: next_self.state = 2;
      2: next_self.state = 0;
    endcase


  end

  simple_dual_port_ram #(
     .DATA_WIDTH(DATA_IN_0_PRECISION_0*DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1),
     .ADDR_WIDTH($clog2(MAX_BUFFER_SIZE)),
     .SIZE      (MAX_BUFFER_SIZE)
  ) ram_inst (
      .clk    (clk),
      .wr_addr(self.write_ptr[ADDR_WIDTH:0]),
      .wr_din (data_in_flattened),
      .wr_en  (ram_wr_en),
      .rd_addr(self.read_ptr[ADDR_WIDTH:0]),
      .rd_dout(ram_rd_dout_flattened)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      self <= '{default: 0};
    end else begin
      self <= next_self;
    end
  end


  assign empty = (self.size == 0);
  assign full  = (self.size == buffer_size);

endmodule