`timescale 1ns / 1ps
module input_buffer #(
    // input
    parameter IN_WIDTH = 8,
    // define as nm * mk
    // rows refers to n, columns refers to m

    //in parallelism in the row dimension
    parameter IN_PARALLELISM = 1,
    parameter IN_SIZE = 8,

    parameter BUFFER_SIZE = 512,
    // this parameter gives out
    // we need to read through all the buffer
    // to be noted here, REPEAT should be included the input stage
    parameter REPEAT = 4,

    parameter OUT_WIDTH = IN_WIDTH,
    parameter OUT_PARALLELISM = IN_PARALLELISM,
    parameter OUT_SIZE = IN_SIZE
) (
    input clk,
    input rst,
    //input data
    input [IN_WIDTH-1:0] data_in[IN_PARALLELISM * IN_SIZE - 1:0],
    input data_in_valid,
    output data_in_ready,

    output [OUT_WIDTH-1:0] data_out[OUT_PARALLELISM * OUT_SIZE - 1:0],
    output data_out_valid,
    input data_out_ready
);

  logic [OUT_WIDTH-1:0] reg_in[OUT_PARALLELISM * OUT_SIZE - 1:0];
  logic reg_in_valid, reg_in_ready;
  localparam COUNTER_WIDTH = $clog2(BUFFER_SIZE * REPEAT);
  logic [COUNTER_WIDTH:0] counter;
  //count output 
  always_ff @(posedge clk) begin
    if (rst) counter <= 0;
    /* verilator lint_off WIDTH */
    else if (reg_in_valid && reg_in_ready)
      if (counter == BUFFER_SIZE * REPEAT - 1) counter <= 0;
      else counter <= counter + 1;
  end

  // clarify a logic to convert 2d to 1d
  logic [IN_WIDTH - 1:0] buffer_out[IN_PARALLELISM * IN_SIZE - 1:0];
  enum {
    STRAIGHT,
    BO
  } mode;
  /* verilator lint_off CMPCONST */
  assign mode = (counter <= BUFFER_SIZE - 1) ? STRAIGHT : BO;
  /* verilator lint_on CMPCONST */
  // BO mode
  //detect the first buffer out, marked as bos(buffer_out_start)
  //delay it, because of the ram delay
  //then pipeline
  logic bos, delay1_bos, delay2_bos;
  assign bos = (reg_in_valid && reg_in_ready) && (counter == BUFFER_SIZE - 1);
  always @(posedge clk) begin
    delay1_bos <= bos;
    delay2_bos <= delay1_bos;
  end
  //signal for ram input  
  localparam ADDR_WIDTH = $clog2(BUFFER_SIZE);
  logic [ADDR_WIDTH:0] addr0;

  //count addr
  //delay_out_valid
  always_ff @(posedge clk) begin
    if (rst) addr0 <= 0;
    /* verilator lint_off WIDTH */
    else if (bos || delay1_bos || delay2_bos)
      if (addr0 == BUFFER_SIZE - 1) addr0 <= 0;
      else addr0 <= addr0 + 1;
    else if (reg_in_valid && reg_in_ready)
      if (addr0 == BUFFER_SIZE - 1) addr0 <= 0;
      else addr0 <= addr0 + 1;
  end

  /* verilator lint_off WIDTH */
  logic ce0;
  logic we0;
  assign ce0 =  (mode == STRAIGHT) ? 1:
                (bos||delay1_bos||delay2_bos)? 1 :
                (reg_in_valid && reg_in_ready)? 1 :0;
  assign we0 = (mode == STRAIGHT) ? data_in_valid && data_in_ready : 0;
  // ram_block #(
  //     .DWIDTH  (IN_WIDTH * IN_PARALLELISM * IN_SIZE),
  //     .AWIDTH  (ADDR_WIDTH),
  //     .MEM_SIZE(BUFFER_SIZE)
  //     /* verilator lint_off PINMISSING */
  // ) ram_buffer (
  //     .addr0(addr0),
  //     .ce0(ce0),
  //     .d0(data_in_flatten),
  //     .we0(we0),
  //     .q0(buffer_out_flatten),
  //     .clk(clk)
  // );
  // when using vivado generate a bram then using this module
  // blk_mem_gen_0 bram (
  //   .clka(clk),
  //   .ena(ce0),
  //   .wea(we0),
  //   .addra(addr0),
  //   .dina(data_in_flatten),
  //   .douta(buffer_out_flatten)
  // );
  //array reshape
  for (genvar i = 0; i < IN_PARALLELISM * IN_SIZE; i++) begin
    blk_mem_gen_0 bram (
        .clka (clk),
        .ena  (ce0),
        .wea  (we0),
        .addra(addr0),
        .dina (data_in[i]),
        .douta(buffer_out[i])
    );
    assign reg_in[i] = (mode == STRAIGHT) ? data_in[i] : buffer_out[i];
  end
  assign reg_in_valid  = (mode == STRAIGHT) ? data_in_valid : (!(delay2_bos || delay1_bos));
  assign data_in_ready = (mode == STRAIGHT) ? reg_in_ready : 0;
  unpacked_skid_buffer #(
      .DATA_WIDTH(OUT_WIDTH),
      .IN_NUM(OUT_PARALLELISM * OUT_SIZE)
  ) reg_inst (
      .data_in(reg_in),
      .data_in_valid(reg_in_valid),
      .data_in_ready(reg_in_ready),
      .*
  );
endmodule

