`ifndef DF_PROCESS_MONITOR__SV
`define DF_PROCESS_MONITOR__SV
`include "sample_agent.sv"
`include "dump_file_agent.sv"
`include "df_process_interface.sv"
class df_process_monitor extends sample_agent;
  virtual df_process_intf in_intf;
  logic [31:0] total_run_time;
  logic [31:0] total_stall_time;
  logic [31:0] read_stall_time;
  logic [31:0] write_stall_time;
  logic [31:0] noContinue_stall_time;
  logic [31:0] noRealStart_stall_time;
  logic [31:0] start_time_arr[$];
  logic [31:0] done_time_arr[$];
  logic [31:0] performance_arr[12];
  dump_file_agent performance_file;

  function new(dump_file_agent file_inst, virtual df_process_intf intf_inst,
               dump_file_agent performance_file_inst);
    super.new(file_inst);
    this.in_intf = intf_inst;
    this.performance_file = performance_file_inst;
    this.total_run_time = 32'h0;
    this.total_stall_time = 32'h0;
    this.read_stall_time = 32'h0;
    this.write_stall_time = 32'h0;
    this.noContinue_stall_time = 32'h0;
    this.noRealStart_stall_time = 32'h0;
    for (integer i = 0; i < 12; i++) this.performance_arr[i] = 32'hffff_ffff;  // initial to -1
  endfunction

  virtual function void output_to_file();
    collect_performance();
    this.performance_file.open_file();
    this.performance_file.dump_1_col(performance_arr);
    while (start_time_arr.size > done_time_arr.size) begin
      done_time_arr.push_back(this.total_run_time);
    end
    this.performance_file.dump_1_line(start_time_arr, done_time_arr);
    this.performance_file.finish_dump();
    this.file_dumper.open_file();
    this.file_dumper.finish_dump();
  endfunction

  virtual task data_monitor();
    fork
      this.record_time();
      this.record_stall();
    join
  endtask

  task record_time();
    logic ready_for_next_start = 1'b1;
    wait (in_intf.reset == 0);
    forever begin
      @(posedge in_intf.clock);
      this.total_run_time = this.total_run_time + 32'h1;
      if (in_intf.ap_start == 1'b1 && ready_for_next_start == 1'b1) begin
        this.start_time_arr.push_back(this.total_run_time);
        ready_for_next_start = 1'b0;
        if (in_intf.ap_ready == 1'b1) ready_for_next_start = 1'b1;
      end else if (in_intf.ap_start == 1'b1 && in_intf.ap_ready == 1'b1)
        ready_for_next_start = 1'b1;
      if (in_intf.ap_done == 1'b1 && in_intf.ap_continue == 1'b1 && this.start_time_arr.size > this.done_time_arr.size)
        this.done_time_arr.push_back(this.total_run_time);
      if (in_intf.pin_stall == 1'b1 || in_intf.pout_stall == 1'b1 || (in_intf.ap_done == 1'b1 && in_intf.ap_continue == 1'b0) || (in_intf.ap_start == 1'b1 && in_intf.real_start == 1'b0)) begin
        this.total_stall_time = this.total_stall_time + 32'h1;
        if (in_intf.pin_stall == 1'b1) this.read_stall_time = this.read_stall_time + 32'h1;
        if (in_intf.pout_stall == 1'b1) this.write_stall_time = this.write_stall_time + 32'h1;
        if (in_intf.ap_done == 1'b1 && in_intf.ap_continue == 1'b0)
          this.noContinue_stall_time = this.noContinue_stall_time + 32'h1;
        if (in_intf.ap_start == 1'b1 && in_intf.real_start == 1'b0)
          this.noRealStart_stall_time = this.noRealStart_stall_time + 32'h1;
      end
      if (in_intf.finish == 1'b1) break;
    end
  endtask

  task record_stall();
    wait (in_intf.reset == 0);
    forever begin
      @(posedge in_intf.clock);
      if (in_intf.finish == 1'b1) break;
    end
  endtask

  function void collect_performance();
    this.performance_arr[0] = this.total_stall_time;
    this.performance_arr[1] = this.total_run_time;
    calc_II();
    calc_latency();
    this.performance_arr[8]  = this.noRealStart_stall_time;  //StallNoStart
    this.performance_arr[9]  = this.noContinue_stall_time;
    this.performance_arr[10] = this.read_stall_time;
    this.performance_arr[11] = this.write_stall_time;
  endfunction

  function void calc_II();
    integer max_II = -1;
    integer min_II = -1;
    integer avg_II = -1;
    integer cur_II = 0;
    integer total_II = 0;
    integer trans_num = this.start_time_arr.size;
    if (trans_num <= 1) begin
      this.performance_arr[3] = max_II;
      this.performance_arr[4] = min_II;
      this.performance_arr[2] = avg_II;
    end else begin
      for (integer i = 1; i < trans_num; i++) begin
        cur_II = this.start_time_arr[i] - this.start_time_arr[i-1];
        if (i == 1) begin
          max_II   = cur_II;
          min_II   = cur_II;
          total_II = cur_II;
        end else begin
          if (cur_II > max_II) max_II = cur_II;
          if (cur_II < min_II) min_II = cur_II;
          total_II = total_II + cur_II;
        end
      end
      avg_II = total_II / (trans_num - 1);
      this.performance_arr[3] = max_II;
      this.performance_arr[4] = min_II;
      this.performance_arr[2] = avg_II;
    end
  endfunction

  function void calc_latency();
    integer max_lat = -1;
    integer min_lat = -1;
    integer avg_lat = -1;
    integer cur_lat = 0;
    integer total_lat = 0;
    integer trans_num = this.start_time_arr.size;
    if (trans_num > this.done_time_arr.size) trans_num = this.done_time_arr.size;
    if (trans_num == 1) begin
      max_lat = this.done_time_arr[0] - this.start_time_arr[0];
      min_lat = this.done_time_arr[0] - this.start_time_arr[0];
      avg_lat = this.done_time_arr[0] - this.start_time_arr[0];
    end else if (trans_num >= 2) begin
      for (integer i = 0; i < trans_num; i++) begin
        cur_lat = this.done_time_arr[i] - this.start_time_arr[i];
        if (i == 0) begin
          max_lat   = cur_lat;
          min_lat   = cur_lat;
          total_lat = cur_lat;
        end else begin
          if (cur_lat > max_lat) max_lat = cur_lat;
          if (cur_lat < min_lat) min_lat = cur_lat;
          total_lat = total_lat + cur_lat;
        end
      end
      avg_lat = total_lat / trans_num;
    end
    this.performance_arr[6] = max_lat;
    this.performance_arr[7] = min_lat;
    this.performance_arr[5] = avg_lat;
  endfunction

endclass
`endif
