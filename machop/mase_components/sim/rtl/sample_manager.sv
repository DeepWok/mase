`ifndef SAMPLE_MANAGER__SV
`define SAMPLE_MANAGER__SV
`include "sample_agent.sv"
class sample_manager;
  sample_agent sample_agent_arr[$];

  function new();
  endfunction

  function void add_one_monitor(sample_agent sample_monitor);
    this.sample_agent_arr.push_back(sample_monitor);
  endfunction

  task start_monitor();
    integer monitor_num = this.sample_agent_arr.size;
    for (integer i = 0; i < monitor_num; i++) begin
      integer j = i;
      fork
        this.sample_agent_arr[j].data_monitor;
      join_none
    end
  endtask

  function void start_dump();
    integer monitor_num = this.sample_agent_arr.size;
    for (integer i = 0; i < monitor_num; i++) this.sample_agent_arr[i].output_to_file;
  endfunction
endclass
`endif
