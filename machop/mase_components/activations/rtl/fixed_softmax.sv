`timescale 1ns / 1ps
module fixed_softmax #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 10,  // input vector size
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,  // 
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,  // incoming elements -
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,  // batch size

    parameter IN_0_DEPTH = $rtoi($ceil(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0)),

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 10,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    parameter OUT_0_DEPTH = $rtoi(
        $ceil(DATA_OUT_0_TENSOR_SIZE_DIM_0 / DATA_OUT_0_PARALLELISM_DIM_0)
    ),

    parameter DATA_INTERMEDIATE_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_INTERMEDIATE_0_PRECISION_1 = DATA_IN_0_PRECISION_1,

    parameter IN_PLACE = 0
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

  logic [DATA_IN_0_PRECISION_0-1:0] ff_data[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] roll_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_INTERMEDIATE_0_PRECISION_0-1:0] exp_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_INTERMEDIATE_0_PRECISION_0-1:0] ff_exp_data[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];

  logic ff_data_valid;
  logic ff_data_ready;

  logic roll_data_valid;
  logic roll_data_ready;

  logic buffer_valid;
  logic buffer_ready;

  logic ff_exp_data_valid;
  logic ff_exp_data_ready;

  localparam SUM_WIDTH = $clog2(DATA_OUT_0_PARALLELISM_DIM_0) + DATA_INTERMEDIATE_0_PRECISION_0;
  localparam ACC_WIDTH = $clog2(OUT_0_DEPTH) + SUM_WIDTH;

  logic [SUM_WIDTH-1:0] summed_exp_data[DATA_OUT_0_PARALLELISM_DIM_1-1:0];  // sum of current block
  logic summed_out_valid[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic summed_out_ready[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic summed_in_ready[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic summed_in_valid;

  logic [ACC_WIDTH-1:0] accumulated_exp_data [DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // accumulation of total vector
  logic [ACC_WIDTH-1:0] ff_accumulated_exp_data [DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // accumulation of total vector

  logic acc_out_valid[DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic acc_out_ready;

  logic ff_acc_valid;
  logic ff_acc_ready;

  unpacked_fifo #(
      .DEPTH(IN_0_DEPTH),
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
  ) roller_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_0),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),  // write enable
      .data_out(ff_data),
      .data_out_valid(ff_data_valid),
      .data_out_ready(ff_data_ready)  // read enable
  );

  localparam STRAIGHT_THROUGH = (DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1 == DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1);

  generate
    if (STRAIGHT_THROUGH) begin
      unpacked_register_slice_quick #(
          .DATA_WIDTH(DATA_IN_0_PRECISION_0),
          .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
      ) single_roll (
          .clk(clk),
          .rst(rst),
          .in_data(ff_data),
          .in_valid(ff_data_valid),
          .in_ready(ff_data_ready),
          .out_data(roll_data),
          .out_valid(roll_data_valid),
          .out_ready(roll_data_ready)
      );

    end else begin

      roller #(
          .DATA_WIDTH(DATA_IN_0_PRECISION_0),
          .NUM(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
          .ROLL_NUM(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1)
      ) roller_inst (
          .clk(clk),
          .rst(rst),
          .data_in(ff_data),
          .data_in_valid(ff_data_valid),
          .data_in_ready(ff_data_ready),
          .data_out(roll_data),
          .data_out_valid(roll_data_valid),
          .data_out_ready(roll_data_ready)
      );
    end
  endgenerate

  split2 #() input_handshake_split (
      .data_in_valid (roll_data_valid),
      .data_in_ready (roll_data_ready),
      .data_out_valid({buffer_valid, summed_in_valid}),
      .data_out_ready({buffer_ready, summed_in_ready[0]})
  );


  for (
      genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1; i++
  ) begin : exp_mem_read
    exp_lut #(
        .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1 (DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_PRECISION_0(DATA_INTERMEDIATE_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_INTERMEDIATE_0_PRECISION_1)
    ) exp_map (
        .data_in_0 (roll_data[i]),
        .data_out_0(exp_data[i])
    );
  end

  unpacked_fifo #(
      .DEPTH(OUT_0_DEPTH),
      .DATA_WIDTH(DATA_INTERMEDIATE_0_PRECISION_0),
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
            .IN_WIDTH(DATA_INTERMEDIATE_0_PRECISION_0)
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

  hold_buffer #(
      .DATA_WIDTH(ACC_WIDTH),
      .DATA_SIZE(DATA_OUT_0_PARALLELISM_DIM_1),
      .DEPTH(OUT_0_DEPTH)
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


  logic [DATA_INTERMEDIATE_0_PRECISION_0 - 1 :0] inter_quotient1 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for rounding division
  logic [DATA_INTERMEDIATE_0_PRECISION_0 - 1 :0] inter_quotient2 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for rounding division
  logic [DATA_INTERMEDIATE_0_PRECISION_0 - 1 :0] inter_quotient3[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for rounding division
  logic [DATA_INTERMEDIATE_0_PRECISION_0 - 1 :0] inter_quotient4 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for rounding division
  logic [DATA_INTERMEDIATE_0_PRECISION_0 + DATA_INTERMEDIATE_0_PRECISION_1 - 1 :0] extended_divisor [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for rounding division

  logic [DATA_INTERMEDIATE_0_PRECISION_0 + DATA_INTERMEDIATE_0_PRECISION_1 - 1 :0] extended_quotient [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for quantization
  logic [DATA_INTERMEDIATE_0_PRECISION_0 - 1 :0] inter_quotient [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0]; // extra bit for quantization



  for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_1; i++) begin : scale_batches
    for (genvar j = 0; j < DATA_OUT_0_PARALLELISM_DIM_0; j++) begin : div_elements
      always_comb begin
        extended_divisor[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] = ff_exp_data[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] << DATA_INTERMEDIATE_0_PRECISION_1;
        extended_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j]  = extended_divisor[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j] / ff_accumulated_exp_data[i];
        inter_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j]  = extended_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j][DATA_INTERMEDIATE_0_PRECISION_0-1:0];
        // data_out_0[DATA_OUT_0_PARALLELISM_DIM_1*(i) + j] = extended_quotient[DATA_OUT_0_PARALLELISM_DIM_1*(i) + j][DATA_OUT_0_PRECISION_0-1:0];
      end
      // quick_round #(
      //   .DATA_WIDTH(DATA_OUT_0_PRECISION_0)
      // ) round (
      //   .data_in(extended_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j][DATA_OUT_0_PRECISION_0-1:1]),
      //   .round_bit(extended_quotient[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j][0]),
      //   .data_out(data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*(i) + j])
      // );
    end
  end
  // assign data_out_0 = inter_quotient;

  // Divide pipeline (retiming)
  logic data_out_0_valid_0;
  logic data_out_0_valid_1;
  logic data_out_0_valid_2;
  logic data_out_0_valid_3;
  logic data_out_0_valid_4;
  always_ff @(posedge clk) begin
    inter_quotient1 <= inter_quotient;
    inter_quotient2 <= inter_quotient1;
    inter_quotient3 <= inter_quotient2;
    inter_quotient4 <= inter_quotient3;

    data_out_0_valid_1 <= data_out_0_valid_0;
    data_out_0_valid_2 <= data_out_0_valid_1;
    data_out_0_valid_3 <= data_out_0_valid_2;
    data_out_0_valid_4 <= data_out_0_valid_3;

  end

  fixed_rounding #(
      .IN_SIZE(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1),
      .IN_WIDTH(DATA_INTERMEDIATE_0_PRECISION_0),
      .IN_FRAC_WIDTH(DATA_INTERMEDIATE_0_PRECISION_1),
      .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  ) data_out_cast (
      .data_in (inter_quotient4),
      .data_out(data_out_0)
  );

  join2 #() output_handshake_split (
      .data_in_valid ({ff_exp_data_valid, ff_acc_valid}),
      .data_in_ready ({ff_exp_data_ready, ff_acc_ready}),
      .data_out_valid(data_out_0_valid_0),
      .data_out_ready(data_out_0_ready)
  );

  assign data_out_0_valid = data_out_0_valid_4;
endmodule

/* verilator lint_off DECLFILENAME */

module hold_buffer #(
    parameter DATA_WIDTH = 16,
    parameter DATA_SIZE = 4,
    parameter DEPTH = 1
) (
    input rst,
    input clk,

    input logic [DATA_WIDTH - 1:0] data_in[DATA_SIZE - 1:0],
    input logic data_in_valid,
    output logic data_in_ready,

    output logic [DATA_WIDTH - 1:0] data_out[DATA_SIZE - 1:0],
    output logic data_out_valid,
    input logic data_out_ready
);

  logic [$clog2(DEPTH) : 0] count;
  logic [ DATA_WIDTH - 1:0] data_out_register[DATA_SIZE - 1:0];
  assign data_out = data_out_register;
  always_ff @(posedge clk) begin
    if (rst) begin
      count <= 0;
      // data_out_register <= 0;
      data_out_valid <= 0;
      data_in_ready <= 1;
    end else begin
      if (count == 0) begin
        // The buffer is empty
        if (data_in_valid) begin
          data_out_register <= data_in;
          count <= DEPTH;
          data_out_valid <= 1;
          data_in_ready <= 0;
        end else begin
          data_in_ready  <= data_out_ready;
          data_out_valid <= 0;
        end
      end else begin
        // The buffer has data
        if (data_out_ready) begin
          count <= count - 1;
        end else begin
          count <= count;
        end
      end
    end
  end

  // take an input and output it for depth length preventing further input from entering. 
endmodule
