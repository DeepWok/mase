import os, logging

from chop.passes.graph.utils import vf, v2p, init_project

logger = logging.getLogger(__name__)


def emit_top_tb(
    tv_dir,
    top_name,
    out_file,
    in_width,
    in_size,
    out_width,
    out_size,
    in_trans_num,
    out_trans_num,
):
    sw_data_in = os.path.join(tv_dir, "sw_data_in.dat")
    sw_data_out = os.path.join(tv_dir, "sw_data_out.dat")
    hw_data_out = os.path.join(tv_dir, "hw_data_out.dat")
    hw_stream_size = os.path.join(tv_dir, "data_in_stream_size.dat")

    buff = f"""
`timescale 1ns / 1ps


`define AUTOTB_DUT {top_name}
`define AUTOTB_DUT_INST AESL_inst_{top_name}
`define AUTOTB_TOP {top_name}_tb
`define AUTOTB_LAT_RESULT_FILE "{top_name}.result.lat.rb"
`define AUTOTB_PER_RESULT_TRANS_FILE "{top_name}.performance.result.transaction.xml"
`define AUTOTB_TOP_INST AESL_inst_apatb_{top_name}_top
`define AUTOTB_MAX_ALLOW_LATENCY 15000000
`define AUTOTB_CLOCK_PERIOD_DIV2 5.00

`define AESL_FIFO_data_in_V AESL_autofifo_data_in_V
`define AESL_FIFO_INST_data_in_V AESL_autofifo_inst_data_in_V
`define AESL_FIFO_data_out_V AESL_autofifo_data_out_V
`define AESL_FIFO_INST_data_out_V AESL_autofifo_inst_data_out_V
`define SW_DATA_IN_DAT "{sw_data_in}"
`define SW_DATA_OUT_DAT "{sw_data_out}"
`define HW_DATA_OUT_DAT "{hw_data_out}"
module `AUTOTB_TOP;

  parameter IN_TRANSACTION_NUM = {in_trans_num};
  parameter OUT_TRANSACTION_NUM = {out_trans_num};
  parameter PROGRESS_TIMEOUT = 10000000;
  parameter LATENCY_ESTIMATION = 0;
  parameter LENGTH_data_in_V = 1;
  parameter LENGTH_data_out_V = 1;
  parameter TOKEN_WIDTH = {max(128, 2*out_width*out_size)+16};
  parameter IN_WIDTH = {in_width};
  parameter IN_SIZE = {in_size};
  parameter OUT_WIDTH = {out_width};
  parameter OUT_SIZE = {out_size};

  task read_token;
    input integer fp;
    output reg [TOKEN_WIDTH-1 : 0] token;
    integer ret;
    begin
      token = "";
      ret   = 0;
      ret   = $fscanf(fp, "%s", token);
    end
  endtask

  task post_check;
    input integer fp1;
    input integer fp2;
    reg [TOKEN_WIDTH-1 : 0] token1;
    reg [TOKEN_WIDTH-1 : 0] token2;
    reg [TOKEN_WIDTH-1 : 0] golden;
    reg [TOKEN_WIDTH-1 : 0] result;
    integer ret;
    begin
      read_token(fp1, token1);
      read_token(fp2, token2);
      if (token1 != "[[[runtime]]]" || token2 != "[[[runtime]]]") begin
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
      end
      read_token(fp1, token1);
      read_token(fp2, token2);
      while (token1 != "[[[/runtime]]]" && token2 != "[[[/runtime]]]") begin
        if (token1 != "[[transaction]]" || token2 != "[[transaction]]") begin
          $display("ERROR: Simulation using HLS TB failed.");
          $finish;
        end
        read_token(fp1, token1);  // skip transaction number
        read_token(fp2, token2);  // skip transaction number
        read_token(fp1, token1);
        read_token(fp2, token2);
        while (token1 != "[[/transaction]]" && token2 != "[[/transaction]]") begin
          ret = $sscanf(token1, "0x%x", golden);
          if (ret != 1) begin
            $display("Failed to parse token!");
            $display("ERROR: Simulation using HLS TB failed.");
            $finish;
          end
          ret = $sscanf(token2, "0x%x", result);
          if (ret != 1) begin
            $display("Failed to parse token!");
            $display("ERROR: Simulation using HLS TB failed.");
            $finish;
          end
          if (golden != result) begin
            $display("%x (expected) vs. %x (actual) - mismatch", golden, result);
            $display("ERROR: Simulation using HLS TB failed.");
            $finish;
          end
          read_token(fp1, token1);
          read_token(fp2, token2);
        end
        read_token(fp1, token1);
        read_token(fp2, token2);
      end
    end
  endtask

  reg AESL_clock;
  reg rst;
  reg dut_rst;
  reg start;
  reg ce;
  reg tb_continue;
  wire AESL_start;
  wire AESL_reset;
  wire AESL_ce;
  wire AESL_ready;
  wire AESL_idle;
  wire AESL_continue;
  wire AESL_done;
  reg AESL_done_delay = 0;
  reg AESL_done_delay2 = 0;
  reg AESL_ready_delay = 0;
  wire ready;
  wire ready_wire;
  wire ap_start;
  wire ap_done;
  wire ap_idle;
  wire ap_ready;
  wire [IN_WIDTH*IN_SIZE-1 : 0] data_in_V_dout;
  wire data_in_V_empty_n;
  wire data_in_V_read;
  wire [OUT_WIDTH*OUT_SIZE-1 : 0] data_out_V_din;
  wire data_out_V_full_n;
  wire data_out_V_write;
  integer done_cnt = 0;
  integer AESL_ready_cnt = 0;
  integer ready_cnt = 0;
  reg ready_initial;
  reg ready_initial_n;
  reg ready_last_n;
  reg ready_delay_last_n;
  reg done_delay_last_n;
  reg interface_done = 0;

  wire ap_clk;
  wire ap_rst;
  wire ap_rst_n;

  wire [IN_WIDTH-1:0] data_in[IN_SIZE-1:0];
  wire [OUT_WIDTH-1:0] data_out[OUT_SIZE-1:0];
  for (genvar i = 0; i < IN_SIZE; i++)
    assign data_in[i] = data_in_V_dout[i*IN_WIDTH+IN_WIDTH-1:i*IN_WIDTH];
  for (genvar i = 0; i < OUT_SIZE; i++)
    assign data_out_V_din[i*OUT_WIDTH+OUT_WIDTH-1:i*OUT_WIDTH] = data_out[i];

  `AUTOTB_DUT `AUTOTB_DUT_INST(
      .clk(ap_clk),
      .rst(ap_rst),
      .data_in(data_in),
      .data_in_valid(data_in_V_empty_n),
      .data_in_ready(data_in_V_read),
      .data_out(data_out),
      .data_out_ready(data_out_V_full_n),
      .data_out_valid(data_out_V_write));

  assign ap_done = data_out_V_write;
  assign ap_ready = data_out_V_write;
  assign ap_idle = ~ap_start;

  // Assignment for control signal
  assign ap_clk = AESL_clock;
  assign ap_rst = dut_rst;
  assign ap_rst_n = ~dut_rst;
  assign AESL_reset = rst;
  assign ap_start = AESL_start;
  assign AESL_start = start;
  assign AESL_done = ap_done;
  assign AESL_idle = ap_idle;
  assign AESL_ready = ap_ready;
  assign AESL_ce = ce;
  assign AESL_continue = tb_continue;
  always @(posedge AESL_clock) begin
    if (AESL_reset) begin
    end else begin
      if (AESL_done !== 1 && AESL_done !== 0) begin
        $display("ERROR: Control signal AESL_done is invalid!");
        $finish;
      end
    end
  end
  always @(posedge AESL_clock) begin
    if (AESL_reset) begin
    end else begin
      if (AESL_ready !== 1 && AESL_ready !== 0) begin
        $display("ERROR: Control signal AESL_ready is invalid!");
        $finish;
      end
    end
  end
  // Fifo Instantiation data_in_V

  wire fifodata_in_V_rd;
  wire [IN_WIDTH*IN_SIZE-1 : 0] fifodata_in_V_dout;
  wire fifodata_in_V_empty_n;
  wire fifodata_in_V_ready;
  wire fifodata_in_V_done;
  reg [31:0] ap_c_n_tvin_trans_num_data_in_V;
  reg data_in_V_ready_reg;

  `AESL_FIFO_data_in_V `AESL_FIFO_INST_data_in_V(
      .clk          (AESL_clock),
      .reset        (AESL_reset),
      .if_write     (),
      .if_din       (),
      .if_full_n    (),
      .if_read      (fifodata_in_V_rd),
      .if_dout      (fifodata_in_V_dout),
      .if_empty_n   (fifodata_in_V_empty_n),
      .ready        (fifodata_in_V_ready),
      .done         (fifodata_in_V_done));

  // Assignment between dut and fifodata_in_V

  // Assign input of fifodata_in_V
  assign fifodata_in_V_rd    = data_in_V_read & data_in_V_empty_n;
  assign fifodata_in_V_ready = data_in_V_ready_reg | ready_initial;
  assign fifodata_in_V_done  = 0;
  // Assign input of dut
  assign data_in_V_dout      = fifodata_in_V_dout;
  reg reg_fifodata_in_V_empty_n;
  initial begin : gen_reg_fifodata_in_V_empty_n_process
    integer proc_rand;
    reg_fifodata_in_V_empty_n = fifodata_in_V_empty_n;
    while (1) begin
      @(fifodata_in_V_empty_n);
      reg_fifodata_in_V_empty_n = fifodata_in_V_empty_n;
    end
  end

  assign data_in_V_empty_n = reg_fifodata_in_V_empty_n;


  //------------------------Fifodata_out_V Instantiation--------------

  // The input and output of fifodata_out_V
  wire fifodata_out_V_wr;
  wire [OUT_SIZE*OUT_WIDTH-1 : 0] fifodata_out_V_din;
  wire fifodata_out_V_full_n;
  wire fifodata_out_V_ready;
  wire fifodata_out_V_done;

  `AESL_FIFO_data_out_V `AESL_FIFO_INST_data_out_V(
      .clk          (AESL_clock),
      .reset        (AESL_reset),
      .if_write     (fifodata_out_V_wr),
      .if_din       (fifodata_out_V_din),
      .if_full_n    (fifodata_out_V_full_n),
      .if_read      (),
      .if_dout      (),
      .if_empty_n   (),
      .ready        (fifodata_out_V_ready),
      .done         (fifodata_out_V_done));

  // Assignment between dut and fifodata_out_V

  // Assign input of fifodata_out_V
  assign fifodata_out_V_wr    = data_out_V_write & data_out_V_full_n;
  assign fifodata_out_V_din   = data_out_V_din;
  assign fifodata_out_V_ready = 0;  //ready_initial | AESL_done_delay;
  assign fifodata_out_V_done  = AESL_done_delay;
  // Assign input of dut
  reg reg_fifodata_out_V_full_n;
  initial begin : gen_reg_fifodata_out_V_full_n_process
    integer proc_rand;
    reg_fifodata_out_V_full_n = fifodata_out_V_full_n;
    while (1) begin
      @(fifodata_out_V_full_n);
      reg_fifodata_out_V_full_n = fifodata_out_V_full_n;
    end
  end

  assign data_out_V_full_n = reg_fifodata_out_V_full_n;


  initial begin : generate_AESL_ready_cnt_proc
    AESL_ready_cnt = 0;
    wait (AESL_reset === 0);
    while (AESL_ready_cnt != OUT_TRANSACTION_NUM) begin
      while (AESL_ready !== 1) begin
        @(posedge AESL_clock);
        #0.4;
      end
      @(negedge AESL_clock);
      AESL_ready_cnt = AESL_ready_cnt + 1;
      @(posedge AESL_clock);
      #0.4;
    end
  end

  event next_trigger_ready_cnt;

  initial begin : gen_ready_cnt
    ready_cnt = 0;
    wait (AESL_reset === 0);
    forever begin
      @(posedge AESL_clock);
      if (ready == 1) begin
        if (ready_cnt < OUT_TRANSACTION_NUM) begin
          ready_cnt = ready_cnt + 1;
        end
      end
      ->next_trigger_ready_cnt;
    end
  end

  wire all_finish = (done_cnt == OUT_TRANSACTION_NUM);

  // done_cnt
  always @(posedge AESL_clock) begin
    if (AESL_reset) begin
      done_cnt <= 0;
    end else begin
      if (AESL_done == 1) begin
        if (done_cnt < OUT_TRANSACTION_NUM) begin
          done_cnt <= done_cnt + 1;
        end
      end
    end
  end

  initial begin : finish_simulation
    integer fp1;
    integer fp2;
    wait (all_finish == 1);
    // last transaction is saved at negedge right after last done
    @(posedge AESL_clock);
    @(posedge AESL_clock);
    @(posedge AESL_clock);
    @(posedge AESL_clock);
    fp1 = $fopen(`SW_DATA_OUT_DAT, "r");
    fp2 = $fopen(`HW_DATA_OUT_DAT, "r");
    if (fp1 == 0)  // Failed to open file
      $display("Failed to open file \\\"%s\\\"", `SW_DATA_OUT_DAT);
    else if (fp2 == 0) $display("Failed to open file \\\"%s\\\"", `HW_DATA_OUT_DAT);
    else begin
      $display(
          "Comparing \\\"%s\\\" with \\\"%s\\\"", `SW_DATA_OUT_DAT, `HW_DATA_OUT_DAT);
      post_check(fp1, fp2);
    end
    $fclose(fp1);
    $fclose(fp2);
    $display("Simulation PASS.");
    $finish;
  end

  initial begin
    AESL_clock = 0;
    forever #`AUTOTB_CLOCK_PERIOD_DIV2 AESL_clock = ~AESL_clock;
  end


  reg end_data_in_V;
  reg [31:0] size_data_in_V;
  reg [31:0] size_data_in_V_backup;
  reg end_data_out_V;
  reg [31:0] size_data_out_V;
  reg [31:0] size_data_out_V_backup;

  initial begin : initial_process
    integer proc_rand;
    rst = 1;
    #100;
    repeat (0 + 3) @(posedge AESL_clock);
    rst = 0;
  end
  initial begin : initial_process_for_dut_rst
    integer proc_rand;
    dut_rst = 1;
    #100;
    repeat (3) @(posedge AESL_clock);
    dut_rst = 0;
  end
  initial begin : start_process
    integer proc_rand;
    reg [31:0] start_cnt;
    ce = 1;
    start = 0;
    start_cnt = 0;
    wait (AESL_reset === 0);
    @(posedge AESL_clock);
    #0 start = 1;
    start_cnt = start_cnt + 1;
    forever begin
      if (start_cnt >= OUT_TRANSACTION_NUM + 1) begin
        #0 start = 0;
      end
      @(posedge AESL_clock);
      if (AESL_ready) begin
        start_cnt = start_cnt + 1;
      end
    end
  end

  always @(AESL_done) begin
    tb_continue = AESL_done;
  end

  initial begin : ready_initial_process
    ready_initial = 0;
    wait (AESL_start === 1);
    ready_initial = 1;
    @(posedge AESL_clock);
    ready_initial = 0;
  end

  always @(posedge AESL_clock) begin
    if (AESL_reset) AESL_ready_delay = 0;
    else AESL_ready_delay = AESL_ready;
  end
  initial begin : ready_last_n_process
    ready_last_n = 1;
    wait (ready_cnt == OUT_TRANSACTION_NUM) @(posedge AESL_clock);
    ready_last_n <= 0;
  end

  always @(posedge AESL_clock) begin
    if (AESL_reset) ready_delay_last_n = 0;
    else ready_delay_last_n <= ready_last_n;
  end
  assign ready = (ready_initial | AESL_ready_delay);
  assign ready_wire = ready_initial | AESL_ready_delay;
  initial begin : done_delay_last_n_process
    done_delay_last_n = 1;
    while (done_cnt < OUT_TRANSACTION_NUM) @(posedge AESL_clock);
    #0.1;
    done_delay_last_n = 0;
  end

  always @(posedge AESL_clock) begin
    if (AESL_reset) begin
      AESL_done_delay  <= 0;
      AESL_done_delay2 <= 0;
    end else begin
      AESL_done_delay  <= AESL_done & done_delay_last_n;
      AESL_done_delay2 <= AESL_done_delay;
    end
  end
  always @(posedge AESL_clock) begin
    if (AESL_reset) interface_done = 0;
    else begin
      #0.01;
      if (ready === 1 && ready_cnt > 0 && ready_cnt < OUT_TRANSACTION_NUM) interface_done = 1;
      else if (AESL_done_delay === 1 && done_cnt == OUT_TRANSACTION_NUM) interface_done = 1;
      else interface_done = 0;
    end
  end
  initial begin : proc_gen_data_in_V_internal_ready
    integer internal_trans_num;
    wait (AESL_reset === 0);
    wait (ready_initial === 1);
    data_in_V_ready_reg <= 0;
    @(posedge AESL_clock);
    internal_trans_num = 1;
    while (internal_trans_num != IN_TRANSACTION_NUM + 1) begin
      if (ap_c_n_tvin_trans_num_data_in_V > internal_trans_num) begin
        data_in_V_ready_reg <= 1;
        @(posedge AESL_clock);
        data_in_V_ready_reg <= 0;
        internal_trans_num = internal_trans_num + 1;
      end else begin
        @(posedge AESL_clock);
      end
    end
    data_in_V_ready_reg <= 0;
  end

  `define STREAM_SIZE_IN_data_in_V "{hw_stream_size}"

  initial begin : gen_ap_c_n_tvin_trans_num_data_in_V
    integer fp_data_in_V;
    reg [TOKEN_WIDTH-1:0] token_data_in_V;
    integer ret;

    ap_c_n_tvin_trans_num_data_in_V = 0;
    end_data_in_V = 0;
    wait (AESL_reset === 0);

    fp_data_in_V = $fopen(`STREAM_SIZE_IN_data_in_V, "r");
    if (fp_data_in_V == 0) begin
      $display("Failed to open file \\\"%s\\\"!", `STREAM_SIZE_IN_data_in_V);
      $finish;
    end
    read_token(fp_data_in_V, token_data_in_V);  // should be [[[runtime]]]
    if (token_data_in_V != "[[[runtime]]]") begin
      $display("ERROR: token_data_in_V != \\\"[[[runtime]]]\\\"");
      $finish;
    end
    size_data_in_V = 0;
    size_data_in_V_backup = 0;
    while (size_data_in_V == 0 && end_data_in_V == 0) begin
      ap_c_n_tvin_trans_num_data_in_V = ap_c_n_tvin_trans_num_data_in_V + 1;
      read_token(fp_data_in_V, token_data_in_V);  // should be [[transaction]] or [[[/runtime]]]
      if (token_data_in_V == "[[transaction]]") begin
        read_token(fp_data_in_V, token_data_in_V);  // should be transaction number
        read_token(fp_data_in_V, token_data_in_V);  // should be size for hls::stream
        ret = $sscanf(token_data_in_V, "%d", size_data_in_V);
        if (size_data_in_V > 0) begin
          size_data_in_V_backup = size_data_in_V;
        end
        read_token(fp_data_in_V, token_data_in_V);  // should be [[/transaction]]
      end else if (token_data_in_V == "[[[/runtime]]]") begin
        $fclose(fp_data_in_V);
        end_data_in_V = 1;
      end else begin
        $display("ERROR: unknown token_data_in_V");
        $finish;
      end
    end
    forever begin
      @(posedge AESL_clock);
      if (end_data_in_V == 0) begin
        if (data_in_V_read == 1 && data_in_V_empty_n == 1) begin
          if (size_data_in_V > 0) begin
            size_data_in_V = size_data_in_V - 1;
            while (size_data_in_V == 0 && end_data_in_V == 0) begin
              ap_c_n_tvin_trans_num_data_in_V = ap_c_n_tvin_trans_num_data_in_V + 1;
              read_token(fp_data_in_V,
                         token_data_in_V);  // should be [[transaction]] or [[[/runtime]]]
              if (token_data_in_V == "[[transaction]]") begin
                read_token(fp_data_in_V, token_data_in_V);  // should be transaction number
                read_token(fp_data_in_V, token_data_in_V);  // should be size for hls::stream
                ret = $sscanf(token_data_in_V, "%d", size_data_in_V);
                if (size_data_in_V > 0) begin
                  size_data_in_V_backup = size_data_in_V;
                end
                read_token(fp_data_in_V, token_data_in_V);  // should be [[/transaction]]
              end else if (token_data_in_V == "[[[/runtime]]]") begin
                size_data_in_V = size_data_in_V_backup;
                $fclose(fp_data_in_V);
                end_data_in_V = 1;
              end else begin
                $display("ERROR: unknown token_data_in_V");
                $finish;
              end
            end
          end
        end
      end else begin
        if (data_in_V_read == 1 && data_in_V_empty_n == 1) begin
          if (size_data_in_V > 0) begin
            size_data_in_V = size_data_in_V - 1;
            if (size_data_in_V == 0) begin
              ap_c_n_tvin_trans_num_data_in_V = ap_c_n_tvin_trans_num_data_in_V + 1;
              size_data_in_V = size_data_in_V_backup;
            end
          end
        end
      end
    end
  end


  reg dump_tvout_finish_data_out_V;

  initial begin : dump_tvout_runtime_sign_data_out_V
    integer fp;
    dump_tvout_finish_data_out_V = 0;
    fp = $fopen(`HW_DATA_OUT_DAT, "w");
    if (fp == 0) begin
      $display("Failed to open file \\\"%s\\\"!", `HW_DATA_OUT_DAT);
      $display("ERROR: Simulation using HLS TB failed.");
      $finish;
    end
    $fdisplay(fp, "[[[runtime]]]");
    $fclose(fp);
    wait (done_cnt == OUT_TRANSACTION_NUM);
    // last transaction is saved at negedge right after last done
    @(posedge AESL_clock);
    @(posedge AESL_clock);
    @(posedge AESL_clock);
    fp = $fopen(`HW_DATA_OUT_DAT, "a");
    if (fp == 0) begin
      $display("Failed to open file \\\"%s\\\"!", `HW_DATA_OUT_DAT);
      $display("ERROR: Simulation using HLS TB failed.");
      $finish;
    end
    $fdisplay(fp, "[[[/runtime]]]");
    $fclose(fp);
    dump_tvout_finish_data_out_V = 1;
  end


  ////////////////////////////////////////////
  // progress and performance
  ////////////////////////////////////////////

  task wait_start();
    while (~AESL_start) begin
      @(posedge AESL_clock);
    end
  endtask

  reg [31:0] clk_cnt = 0;
  reg AESL_ready_p1;
  reg AESL_start_p1;

  always @(posedge AESL_clock) begin
    if (AESL_reset == 1) begin
      clk_cnt <= 32'h0;
      AESL_ready_p1 <= 1'b0;
      AESL_start_p1 <= 1'b0;
    end else begin
      clk_cnt <= clk_cnt + 1;
      AESL_ready_p1 <= AESL_ready;
      AESL_start_p1 <= AESL_start;
    end
  end

  reg [31:0] start_timestamp[0:OUT_TRANSACTION_NUM - 1];
  reg [31:0] start_cnt;
  reg [31:0] ready_timestamp[0:OUT_TRANSACTION_NUM - 1];
  reg [31:0] ap_ready_cnt;
  reg [31:0] finish_timestamp[0:OUT_TRANSACTION_NUM - 1];
  reg [31:0] finish_cnt;
  reg [31:0] lat_total;
  event report_progress;

  always @(posedge AESL_clock) begin
    if (finish_cnt == OUT_TRANSACTION_NUM - 1 && AESL_done == 1'b1)
      lat_total = clk_cnt - start_timestamp[0];
  end

  initial begin
    start_cnt = 0;
    finish_cnt = 0;
    ap_ready_cnt = 0;
    wait (AESL_reset == 0);
    wait_start();
    start_timestamp[start_cnt] = clk_cnt;
    start_cnt = start_cnt + 1;
    if (AESL_done) begin
      finish_timestamp[finish_cnt] = clk_cnt;
      finish_cnt = finish_cnt + 1;
    end
    ->report_progress;
    forever begin
      @(posedge AESL_clock);
      if (start_cnt < OUT_TRANSACTION_NUM) begin
        if ((AESL_start && AESL_ready_p1) || (AESL_start && ~AESL_start_p1)) begin
          start_timestamp[start_cnt] = clk_cnt;
          start_cnt = start_cnt + 1;
        end
      end
      if (ap_ready_cnt < OUT_TRANSACTION_NUM) begin
        if (AESL_start_p1 && AESL_ready_p1) begin
          ready_timestamp[ap_ready_cnt] = clk_cnt;
          ap_ready_cnt = ap_ready_cnt + 1;
        end
      end
      if (finish_cnt < OUT_TRANSACTION_NUM) begin
        if (AESL_done) begin
          finish_timestamp[finish_cnt] = clk_cnt;
          finish_cnt = finish_cnt + 1;
        end
      end
      ->report_progress;
    end
  end

  reg [31:0] progress_timeout;

  initial begin : simulation_progress
    real intra_progress;
    wait (AESL_reset == 0);
    progress_timeout = PROGRESS_TIMEOUT;
    $display(
        "////////////////////////////////////////////////////////////////////////////////////");
    $display("// Inter-Transaction Progress: Completed Transaction / Total Transaction");
    $display("// Intra-Transaction Progress: Measured Latency / Latency Estimation * 100%%");
    $display("//");
    $display(
        "// RTL Simulation : \\\"Inter-Transaction Progress\\\" [\\\"Intra-Transaction Progress\\\"] @ \\\"Simulation Time\\\"");
    $display(
        "////////////////////////////////////////////////////////////////////////////////////");
    print_progress();
    while (finish_cnt < OUT_TRANSACTION_NUM) begin
      @(report_progress);
      if (finish_cnt < OUT_TRANSACTION_NUM) begin
        if (AESL_done) begin
          print_progress();
          progress_timeout = PROGRESS_TIMEOUT;
        end else begin
          if (progress_timeout == 0) begin
            print_progress();
            progress_timeout = PROGRESS_TIMEOUT;
          end else begin
            progress_timeout = progress_timeout - 1;
          end
        end
      end
    end
    print_progress();
    $display(
        "////////////////////////////////////////////////////////////////////////////////////");
    calculate_performance();
  end

  task get_intra_progress(output real intra_progress);
    begin
      if (start_cnt > finish_cnt) begin
        intra_progress = clk_cnt - start_timestamp[finish_cnt];
      end else if (finish_cnt > 0) begin
        intra_progress = LATENCY_ESTIMATION;
      end else begin
        intra_progress = 0;
      end
      intra_progress = intra_progress / LATENCY_ESTIMATION;
    end
  endtask

  task print_progress();
    real intra_progress;
    begin
      if (LATENCY_ESTIMATION > 0) begin
        get_intra_progress(intra_progress);
        $display("// RTL Simulation : %0d / %0d [%2.2f%%] @ \\\"%0t\\\"", finish_cnt,
                 OUT_TRANSACTION_NUM, intra_progress * 100, $time);
      end else begin
        $display("// RTL Simulation : %0d / %0d [n/a] @ \\\"%0t\\\"", finish_cnt,
                 OUT_TRANSACTION_NUM, $time);
      end
    end
  endtask

  task calculate_performance();
    integer i;
    integer fp;
    reg [31:0] latency[0:OUT_TRANSACTION_NUM - 1];
    reg [31:0] latency_min;
    reg [31:0] latency_max;
    reg [31:0] latency_total;
    reg [31:0] latency_average;
    reg [31:0] interval[0:OUT_TRANSACTION_NUM - 2];
    reg [31:0] interval_min;
    reg [31:0] interval_max;
    reg [31:0] interval_total;
    reg [31:0] interval_average;
    reg [31:0] total_execute_time;
    begin
      latency_min = -1;
      latency_max = 0;
      latency_total = 0;
      interval_min = -1;
      interval_max = 0;
      interval_total = 0;
      total_execute_time = lat_total;

      for (i = 0; i < OUT_TRANSACTION_NUM; i = i + 1) begin
        // calculate latency
        latency[i] = finish_timestamp[i] - start_timestamp[i];
        if (latency[i] > latency_max) latency_max = latency[i];
        if (latency[i] < latency_min) latency_min = latency[i];
        latency_total = latency_total + latency[i];
        // calculate interval
        if (OUT_TRANSACTION_NUM == 1) begin
          interval[i] = 0;
          interval_max = 0;
          interval_min = 0;
          interval_total = 0;
        end else if (i < OUT_TRANSACTION_NUM - 1) begin
          interval[i] = start_timestamp[i+1] - start_timestamp[i];
          if (interval[i] > interval_max) interval_max = interval[i];
          if (interval[i] < interval_min) interval_min = interval[i];
          interval_total = interval_total + interval[i];
        end
      end

      latency_average = latency_total / OUT_TRANSACTION_NUM;
      if (OUT_TRANSACTION_NUM == 1) begin
        interval_average = 0;
      end else begin
        interval_average = interval_total / (OUT_TRANSACTION_NUM - 1);
      end

      fp = $fopen(`AUTOTB_LAT_RESULT_FILE, "w");

      $fdisplay(fp, "$MAX_LATENCY = \\\"%0d\\\"", latency_max);
      $fdisplay(fp, "$MIN_LATENCY = \\\"%0d\\\"", latency_min);
      $fdisplay(fp, "$AVER_LATENCY = \\\"%0d\\\"", latency_average);
      $fdisplay(fp, "$MAX_THROUGHPUT = \\\"%0d\\\"", interval_max);
      $fdisplay(fp, "$MIN_THROUGHPUT = \\\"%0d\\\"", interval_min);
      $fdisplay(fp, "$AVER_THROUGHPUT = \\\"%0d\\\"", interval_average);
      $fdisplay(fp, "$TOTAL_EXECUTE_TIME = \\\"%0d\\\"", total_execute_time);

      $fclose(fp);

      fp = $fopen(`AUTOTB_PER_RESULT_TRANS_FILE, "w");

      $fdisplay(fp, "%20s%16s%16s", "", "latency", "interval");
      if (OUT_TRANSACTION_NUM == 1) begin
        i = 0;
        $fdisplay(fp, "transaction%8d:%16d%16d", i, latency[i], interval[i]);
      end else begin
        for (i = 0; i < OUT_TRANSACTION_NUM; i = i + 1) begin
          if (i < OUT_TRANSACTION_NUM - 1) begin
            $fdisplay(fp, "transaction%8d:%16d%16d", i, latency[i], interval[i]);
          end else begin
            $fdisplay(fp, "transaction%8d:%16d               x", i, latency[i]);
          end
        end
      end

      $fclose(fp);
    end
  endtask


  ////////////////////////////////////////////
  // Dependence Check
  ////////////////////////////////////////////

`ifndef POST_SYN

`endif
  ///////////////////////////////////////////////////////
  // dataflow status monitor
  ///////////////////////////////////////////////////////
  // dataflow_monitor U_dataflow_monitor (
  //     .clock (AESL_clock),
  //     .reset (rst),
  //     .finish(all_finish)
  // );

  // `include "fifo_para.v"

endmodule
"""

    with open(out_file, "w", encoding="utf-8") as outf:
        outf.write(buff)
    logger.debug(f"Top-level test bench emitted to {out_file}")
    assert os.path.isfile(out_file), "Emitting top-level test bench failed."
    os.system(f"verible-verilog-format --inplace {out_file}")
