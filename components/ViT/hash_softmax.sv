`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
module hash_softmax #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter EXP_WIDTH = 8,
    parameter EXP_FRAC_WIDTH = 4,
    parameter DIV_WIDTH = 8,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 4,
    parameter IN_SIZE = 4,
    parameter OUT_SIZE = 2,
    parameter IN_DEPTH = 3
) (
    input clk,
    rst,
    input [IN_WIDTH - 1:0] data_in[IN_SIZE - 1:0],
    input data_in_valid,
    output data_in_ready,
    output [OUT_WIDTH - 1:0] data_out[OUT_SIZE - 1:0],
    output data_out_valid,
    input data_out_ready
);
  logic [EXP_WIDTH - 1:0] exp[IN_SIZE - 1:0];
  logic exp_valid, exp_ready;
  for (genvar i = 0; i < IN_SIZE; i++) begin : exp_parallel
    /* verilator lint_off UNUSEDSIGNAL */
    logic exp_in_valid, exp_in_ready;
    logic exp_out_valid, exp_out_ready;
    assign exp_out_ready = exp_ready;
    assign exp_in_valid  = data_in_valid;
    hash_exp #(
        .IN_WIDTH (IN_WIDTH),
        .OUT_WIDTH(EXP_WIDTH)
    ) exp_inst (
        .data_in(data_in[i]),
        .data_in_valid(exp_in_valid),
        .data_in_ready(exp_in_ready),
        .data_out(exp[i]),
        .data_out_valid(exp_out_valid),
        .data_out_ready(exp_out_ready),
        .*
    );
  end
  assign exp_valid = exp_parallel[0].exp_out_valid;
  assign data_in_ready = exp_parallel[0].exp_in_ready;
  localparam SUM_WIDTH = EXP_WIDTH + $clog2(IN_SIZE);
  logic [SUM_WIDTH - 1:0] sum;
  logic sum_valid, sum_ready;
  logic exp2sum_ready, exp2roller_ready;
  logic exp2sum_valid, exp2roller_valid;

  split2 split2_inst (
      .data_in_valid (exp_valid),
      .data_in_ready (exp_ready),
      .data_out_valid({exp2sum_valid, exp2roller_valid}),
      .data_out_ready({exp2sum_ready, exp2roller_ready})
  );

  fixed_adder_tree #(
      .IN_SIZE (IN_SIZE),
      .IN_WIDTH(IN_WIDTH)
  ) sum_inst (
      .data_in(exp),
      .data_in_valid(exp2sum_valid),
      .data_in_ready(exp2sum_ready),
      .data_out(sum),
      .data_out_valid(sum_valid),
      .data_out_ready(sum_ready),
      .*
  );
  // roller part
  logic [EXP_WIDTH - 1:0] roller_exp[OUT_SIZE - 1:0];
  logic roller_exp_valid, roller_exp_ready;
  logic [EXP_WIDTH - 1:0] ff_exp[IN_SIZE - 1:0];
  logic ff_exp_valid, ff_exp_ready;
  unpacked_fifo #(
      .DEPTH(IN_DEPTH),
      .DATA_WIDTH(EXP_WIDTH),
      .IN_NUM(IN_SIZE)
  ) roller_buffer (
      .data_in(exp),
      .data_in_valid(exp2roller_valid),
      .data_in_ready(exp2roller_ready),
      .data_out(ff_exp),
      .data_out_valid(ff_exp_valid),
      .data_out_ready(ff_exp_ready),
      .*
  );
  roller #(
      .DATA_WIDTH(EXP_WIDTH),
      .NUM(IN_SIZE),
      .IN_SIZE(IN_SIZE),
      .ROLL_NUM(OUT_SIZE)
  ) roller_inst (
      .data_in(ff_exp),
      .data_in_valid(ff_exp_valid),
      .data_in_ready(ff_exp_ready),
      .data_out(roller_exp),
      .data_out_valid(roller_exp_valid),
      .data_out_ready(roller_exp_ready),
      .*
  );

  localparam ACC_WIDTH = SUM_WIDTH + $clog2(IN_DEPTH);
  logic [ACC_WIDTH - 1:0] acc;
  logic [ACC_WIDTH - 1:0] acc_duplicate[OUT_SIZE - 1:0];
  logic acc_valid, acc_ready;

  fixed_accumulator #(
      .IN_WIDTH(SUM_WIDTH),
      .IN_DEPTH(IN_DEPTH)
  ) fixed_accumulator_inst (
      .clk(clk),
      .rst(rst),
      .data_in(sum),
      .data_in_valid(sum_valid),
      .data_in_ready(sum_ready),
      .data_out(acc),
      .data_out_valid(acc_valid),
      .data_out_ready(acc_ready)
  );

  logic [ACC_WIDTH - 1:0] ib_acc[OUT_SIZE - 1:0];
  logic ib_acc_valid, ib_acc_ready;
  circular_buffer #(
      .IN_WIDTH(ACC_WIDTH),
      .IN_SIZE (OUT_SIZE),
      .REPEAT  (IN_DEPTH * IN_SIZE / OUT_SIZE)
  ) acc_circular (
      .clk(clk),
      .rst(rst),
      .data_in(acc_duplicate),
      .data_in_valid(acc_valid),
      .data_in_ready(acc_ready),
      .data_out(ib_acc),
      .data_out_valid(ib_acc_valid),
      .data_out_ready(ib_acc_ready)
  );

  logic [ACC_WIDTH - 1:0] one_over_div[OUT_SIZE - 1:0];
  logic [DIV_WIDTH - 1:0] div_in[OUT_SIZE - 1:0];

  fixed_rounding #(
      .IN_SIZE(OUT_SIZE),
      .IN_WIDTH(ACC_WIDTH),
      .IN_FRAC_WIDTH(0),
      .OUT_WIDTH(DIV_WIDTH),
      .OUT_FRAC_WIDTH(0)
  ) div_round (
      .data_in (one_over_div),
      .data_out(div_in)
  );
  logic div_join_valid, div_join_ready;

  join2 #() div_join_inst (
      .data_in_ready ({roller_exp_ready, ib_acc_ready}),
      .data_in_valid ({roller_exp_valid, ib_acc_valid}),
      .data_out_valid(div_join_valid),
      .data_out_ready(div_join_ready)
  );
  for (genvar i = 0; i < OUT_SIZE; i++) begin : div_parallel
    /* verilator lint_off UNUSEDSIGNAL */
    logic [OUT_WIDTH - 1:0] div_out;
    logic div_out_valid, div_out_ready;
    assign acc_duplicate[i] = acc;
    /* verilator lint_off WIDTH */
    assign one_over_div[i] = (roller_exp[i] == 0) ? 2**(DIV_WIDTH-1) - 1:ib_acc[i] / roller_exp[i];
    /* verilator lint_on WIDTH */
    logic div_in_valid, div_in_ready;
    assign div_in_valid = div_join_valid;
    hash_div #(
        .IN_WIDTH (DIV_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) div_inst (
        .data_in(div_in[i]),
        .data_in_valid(div_in_valid),
        .data_in_ready(div_in_ready),
        .data_out(div_out),
        .data_out_valid(div_out_valid),
        .data_out_ready(div_out_ready),
        .*
    );
    assign data_out[i]   = div_out;
    assign div_out_ready = data_out_ready;
  end
  assign div_join_ready = div_parallel[0].div_in_ready;
  assign data_out_valid = div_parallel[0].div_out_valid;
endmodule
/* verilator lint_off DECLFILENAME */

module hash_exp #(
    parameter IN_WIDTH  = 8,
    parameter OUT_WIDTH = 8
) (
    input clk,
    input rst,
    input [IN_WIDTH - 1:0] data_in,
    input data_in_valid,
    output data_in_ready,
    output logic [OUT_WIDTH - 1:0] data_out,
    output data_out_valid,
    input data_out_ready
);
  localparam MEM_NUM = 2 ** (IN_WIDTH + 1) - 1;
  logic [OUT_WIDTH - 1:0] mem[MEM_NUM - 1:0];
  initial begin
    $readmemh("../exp_init.mem", mem);
  end

  // The shift register stores the validity of the data in the buffer 
  logic shift_reg;
  // The buffer stores the intermeidate data being computed in the register slice
  logic [OUT_WIDTH-1:0] buffer;

  always_ff @(posedge clk) begin
    if (rst) shift_reg <= 1'b0;
    else begin
      // no backpressure or buffer empty
      if (data_out_ready || !shift_reg) shift_reg <= data_in_valid;
      else shift_reg <= shift_reg;
    end
  end

  // buffer 
  always_ff @(posedge clk) begin
    if (rst) buffer <= 0;
    // backpressure && valid output
    if (!data_out_ready && data_out_valid) buffer <= buffer;
    /* verilator lint_off WIDTH */
    else
      buffer <= mem[data_in];
    /* verilator lint_on WIDTH */
  end

  // empty buffer or no back pressure 
  assign data_in_ready = (~shift_reg) | data_out_ready;
  // dummy data_iniring 
  assign data_out_valid = shift_reg;
  assign data_out = buffer;
endmodule


module hash_div #(
    parameter IN_WIDTH  = 8,
    parameter OUT_WIDTH = 8
) (
    input clk,
    input rst,
    input [IN_WIDTH - 1:0] data_in,
    input data_in_valid,
    output logic data_in_ready,
    output logic [OUT_WIDTH - 1:0] data_out,
    output logic data_out_valid,
    input data_out_ready
);
  localparam MEM_NUM = 2 ** (IN_WIDTH + 1) - 1;
  logic [OUT_WIDTH - 1:0] mem[MEM_NUM - 1:0];
  initial begin
    $readmemh("../div_init.mem", mem);
  end

  // The shift register stores the validity of the data in the buffer 
  logic shift_reg;
  // The buffer stores the intermeidate data being computed in the register slice
  logic [OUT_WIDTH-1:0] buffer;

  always_ff @(posedge clk) begin
    if (rst) shift_reg <= 1'b0;
    else begin
      // no backpressure or buffer empty
      if (data_out_ready || !shift_reg) shift_reg <= data_in_valid;
      else shift_reg <= shift_reg;
    end
  end

  // buffer
  always_ff @(posedge clk) begin
    if (rst) buffer <= 0;
    // backpressure && valid output
    if (!data_out_ready && data_out_valid) buffer <= buffer;
    /* verilator lint_off WIDTH */
    else
      buffer <= mem[data_in];
    /* verilator lint_on WIDTH */
  end

  // empty buffer or no back pressure 
  assign data_in_ready = (~shift_reg) | data_out_ready;
  // dummy data_iniring 
  assign data_out_valid = shift_reg;
  assign data_out = buffer;
endmodule

module circular_buffer #(
    // input
    parameter IN_WIDTH = 8,
    // define as nm * mk
    // rows refers to n, columns refers to m

    //in parallelism in the row dimension
    parameter IN_SIZE = 1,
    parameter REPEAT = 8,
    parameter OUT_WIDTH = IN_WIDTH,
    parameter OUT_SIZE = IN_SIZE
) (
    input clk,
    input rst,
    //input data
    input [IN_WIDTH-1:0] data_in[IN_SIZE - 1:0],
    input logic data_in_valid,
    output logic data_in_ready,

    output [OUT_WIDTH-1:0] data_out[OUT_SIZE - 1:0],
    output logic data_out_valid,
    input logic data_out_ready
);
  localparam COUNT_SIZE = $clog2(REPEAT) + 1;
  logic [IN_WIDTH-1:0] buffer[IN_SIZE - 1:0];
  // The shift register stores the validity of the data in the buffer 
  logic circular_mode;
  logic [COUNT_SIZE - 1:0] circular_count;
  assign circular_mode = circular_count != 0;
  logic insert, remove;
  always_comb begin
    insert = data_in_ready && data_in_valid;
    remove = data_out_ready && data_out_valid;
  end

  always_ff @(posedge clk) begin
    /* verilator lint_off WIDTH */
    if (rst) circular_count <= 0;
    else if (insert && (~circular_mode)) circular_count <= REPEAT;
    else if (remove && circular_mode) circular_count <= circular_count - 1;
    else circular_count <= circular_count;
    /* verilator lint_on WIDTH */
  end
  for (genvar i = 0; i < IN_SIZE; i++) begin
    always_ff @(posedge clk) begin
      if (rst) buffer[i] <= 0;
      else if (insert && (~circular_mode)) buffer[i] <= data_in[i];
      else buffer[i] <= buffer[i];
    end
    assign data_out[i] = buffer[i];
  end

  always_comb begin
    // empty buffer or no back pressure
    data_in_ready  = (~circular_mode);
    // dummy data_iniring
    data_out_valid = circular_mode;
  end
endmodule

