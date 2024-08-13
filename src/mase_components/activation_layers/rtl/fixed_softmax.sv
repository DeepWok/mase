`timescale 1ns / 1ps
module fixed_softmax #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,  // input vector size
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 6,  // 
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 3,  // incoming elements -
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,  // batch size

    parameter IN_0_DEPTH = $rtoi($ceil(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)),

    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_PRECISION_0 = DATA_OUT_0_PRECISION_1 + 2,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,

    parameter OUT_0_DEPTH = IN_0_DEPTH,

    parameter DATA_EXP_0_PRECISION_0 = 12,
    parameter DATA_EXP_0_PRECISION_1 = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  // softmax over a vector
  // each vector might be split into block of elements
  // Can handle multiple batches at once
  // each iteration recieves a batch of blocks

  logic [DATA_IN_0_PRECISION_0-1:0] roll_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_EXP_0_PRECISION_0-1:0] exp_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_EXP_0_PRECISION_0-1:0] ff_exp_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];

  // logic ff_data_valid;
  // logic ff_data_ready;

  logic roll_data_valid;
  logic roll_data_ready;

  logic buffer_valid;
  logic buffer_ready;

  logic ff_exp_data_valid;
  logic ff_exp_data_ready;

  localparam SUM_WIDTH = $clog2(DATA_OUT_0_PARALLELISM_DIM_0) + DATA_EXP_0_PRECISION_0;
  localparam ACC_WIDTH = $clog2(OUT_0_DEPTH) + SUM_WIDTH;

  logic [SUM_WIDTH-1:0] summed_exp_data[DATA_OUT_0_PARALLELISM_DIM_1-1:0];  // sum of current block
  logic summed_out_valid[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic summed_out_ready[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic summed_in_ready[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic summed_in_valid;

  logic [ACC_WIDTH-1:0] accumulated_exp_data [DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // accumulation of total vector
  logic [ACC_WIDTH-1:0] ff_accumulated_exp_data [DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // accumulation of total vector
  logic [ACC_WIDTH-1:0] ff_accumulated_exp_data_dup [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // duplication accumulation of total vector
  logic acc_out_valid[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic acc_out_ready;

  logic ff_acc_valid;
  logic ff_acc_ready;


  split2 #() input_handshake_split (
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out_valid({buffer_valid, summed_in_valid}),
      .data_out_ready({buffer_ready, summed_in_ready[0]})
  );


  for (
      genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1; i++
  ) begin : exp_mem_read
    exp_lut #(
        .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1 (DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_PRECISION_0(DATA_EXP_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_EXP_0_PRECISION_1)
    ) exp_map (
        .data_in_0 (data_in_0[i]),
        .data_out_0(exp_data[i])
    );
  end

  unpacked_fifo #(
      .DEPTH(OUT_0_DEPTH*8),
      .DATA_WIDTH(DATA_EXP_0_PRECISION_0),
      .IN_NUM(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
  ) out_roller_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(exp_data),
      .data_in_valid(buffer_valid),
      .data_in_ready(buffer_ready),  // write enable
      .data_out(ff_exp_data),
      .data_out_valid(ff_exp_data_valid),
      .data_out_ready(ff_exp_data_ready)  // read enable
  );


  generate
    for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : accumulate_batches
      if (DATA_OUT_0_PARALLELISM_DIM_0 > 1) begin
        fixed_adder_tree #(
            .IN_SIZE (DATA_OUT_0_PARALLELISM_DIM_0),
            .IN_WIDTH(DATA_EXP_0_PRECISION_0)
        ) block_sum (
            .clk(clk),
            .rst(rst),
            .data_in(exp_data[DATA_OUT_0_PARALLELISM_DIM_0*i+:DATA_OUT_0_PARALLELISM_DIM_0]),
            .data_in_valid(summed_in_valid),  // adder enable
            .data_in_ready(summed_in_ready[i]), // addition complete - need to join with the buffer ready and many readys
            .data_out(summed_exp_data[i]),  // create a sum variable for the mini set 
            .data_out_valid(summed_out_valid[i]),  // sum is valid
            .data_out_ready(summed_out_ready[i])  // next module needs the sum 
        );

      end else begin
        assign summed_exp_data[i]  = exp_data[i];  // DATA_OUT_PLL_0 == 1
        assign summed_out_valid[i] = summed_in_valid;
        assign summed_in_ready[i]  = summed_out_ready[i];
      end



      fixed_accumulator #(
          .IN_WIDTH(SUM_WIDTH),
          .IN_DEPTH(OUT_0_DEPTH)
      ) fixed_accumulator_inst (
          .clk(clk),
          .rst(rst),
          .data_in(summed_exp_data[i]),  // sum variable for mini set
          .data_in_valid(summed_out_valid[i]),  // accumulator enable
          .data_in_ready(summed_out_ready[i]),  // accumulator complete
          .data_out(accumulated_exp_data[i]),  // accumulated variable
          .data_out_valid(acc_out_valid[i]), //accumulation of ALL variables complete (this is my state machine)
          .data_out_ready(acc_out_ready)  // Start the accumulation
      );
    end
  endgenerate

  input_buffer #(
      .DATA_WIDTH(ACC_WIDTH),
      .IN_NUM(DATA_OUT_0_PARALLELISM_DIM_1),
      .BUFFER_SIZE(1),
      .REPEAT(IN_0_DEPTH)
  ) acc_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(accumulated_exp_data),
      .data_in_valid(acc_out_valid[0]),
      .data_in_ready(acc_out_ready),  // write enable
      .data_out(ff_accumulated_exp_data),
      .data_out_valid(ff_acc_valid),
      .data_out_ready(ff_acc_ready)  // read enable
  );

  //TODO: change to register slice

  logic [DATA_EXP_0_PRECISION_0 + DATA_OUT_0_PRECISION_1 - 1 :0] extended_divisor [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for rounding division
  logic [DATA_OUT_0_PRECISION_0 + DATA_OUT_0_PRECISION_1 - 1 :0] extended_quotient [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for quantization

  for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : scale_batches
    for (genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : div_elements
      always_comb begin
        extended_divisor[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] = ff_exp_data[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] << DATA_OUT_0_PRECISION_1;
        ff_accumulated_exp_data_dup[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] = ff_accumulated_exp_data[i];
        // extended_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] = extended_divisor[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] / ff_accumulated_exp_data[i];
        data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j]  = extended_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j][DATA_OUT_0_PRECISION_0-1:0];
      end
    end
  end

  // join2 #() output_handshake_split (
  //     .data_in_valid ({ff_exp_data_valid, ff_acc_valid}),
  //     .data_in_ready ({ff_exp_data_ready, ff_acc_ready}),
  //     .data_out_valid(data_out_0_valid),
  //     .data_out_ready(data_out_0_ready)
  // );
  fixed_div #(
      .IN_NUM(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1),
      .DIVIDEND_WIDTH(DATA_EXP_0_PRECISION_0 + DATA_OUT_0_PRECISION_1),
      .DIVISOR_WIDTH(ACC_WIDTH),
      .QUOTIENT_WIDTH(DATA_OUT_0_PRECISION_0 + DATA_OUT_0_PRECISION_1)
  ) div_inst (
      .clk(clk),
      .rst(rst),
      .dividend_data(extended_divisor),
      .dividend_data_valid(ff_exp_data_valid),
      .dividend_data_ready(ff_exp_data_ready),
      .divisor_data(ff_accumulated_exp_data_dup),
      .divisor_data_valid(ff_acc_valid),
      .divisor_data_ready(ff_acc_ready),
      .quotient_data(extended_quotient),
      .quotient_data_valid(data_out_0_valid),
      .quotient_data_ready(data_out_0_ready)
  );
endmodule
