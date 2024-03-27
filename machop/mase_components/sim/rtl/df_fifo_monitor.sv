`ifndef DF_FIFO_MONITOR__SV
`define DF_FIFO_MONITOR__SV
`include "sample_agent.sv"
`include "dump_file_agent.sv"
`include "df_fifo_interface.sv"
class df_fifo_monitor extends sample_agent;
  virtual df_fifo_intf in_intf;
  logic [31:0] rt_depth;
  logic chan_status_wb;
  logic chan_status_rb;
  logic [31:0] total_run_time;
  logic [31:0] total_wb_time;
  logic [31:0] total_rb_time;
  logic [31:0] max_used_depth;
  dump_file_agent chan_file;

  function new(dump_file_agent file_inst, virtual df_fifo_intf intf_inst,
               dump_file_agent chan_file_inst);
    super.new(file_inst);
    this.in_intf = intf_inst;
    this.chan_file = chan_file_inst;
    this.rt_depth = 32'h0;
    this.chan_status_wb = 1'b0;
    this.chan_status_rb = 1'b0;
    this.total_wb_time = 32'h0;
    this.total_rb_time = 32'h0;
    this.total_run_time = 32'h0;
    this.max_used_depth = 32'h0;
  endfunction

  virtual function void output_to_file();
    logic [31:0] chan_status[5];
    chan_status[0] = {30'h0, this.chan_status_rb, this.chan_status_wb};
    chan_status[1] = this.total_wb_time;
    chan_status[2] = this.total_rb_time;
    chan_status[3] = this.total_run_time;
    chan_status[4] = this.max_used_depth;
    this.chan_file.open_file();
    this.chan_file.dump_1_col(chan_status);
    this.chan_file.finish_dump();
    this.file_dumper.open_file();
    this.file_dumper.finish_dump();
  endfunction

  virtual task data_monitor();
    fork
      this.record_depth();
      this.record_block();
    join
  endtask

  task record_depth();
    wait (in_intf.reset == 0);
    forever begin
      @(posedge in_intf.clock);
      if (in_intf.wr_en == 1'b1 && in_intf.rd_en == 1'b1) this.rt_depth = this.rt_depth;
      else if (in_intf.wr_en == 1'b1) this.rt_depth = this.rt_depth + 32'h1;
      else if (in_intf.rd_en == 1'b1 && this.rt_depth > 32'h0)
        this.rt_depth = this.rt_depth - 32'h1;
      if (this.rt_depth > this.max_used_depth) this.max_used_depth = this.rt_depth;
      if (in_intf.finish == 1) break;
    end
  endtask

  task record_block();
    wait (in_intf.reset == 0);
    forever begin
      @(posedge in_intf.clock);
      this.total_run_time = this.total_run_time + 32'h1;
      if (in_intf.fifo_wr_block == 1) begin
        this.chan_status_wb = 1'b1;
        this.total_wb_time  = this.total_wb_time + 32'h1;
      end else if (in_intf.fifo_rd_block == 1) begin
        this.chan_status_rb = 1'b1;
        this.total_rb_time  = this.total_rb_time + 32'h1;
      end else if (in_intf.finish == 1) break;
    end
  endtask

endclass
`endif
