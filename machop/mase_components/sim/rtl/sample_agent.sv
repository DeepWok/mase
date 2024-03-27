// base class for dataflow sample monitors
`ifndef SAMPLE_AGENT__SV
`define SAMPLE_AGENT__SV
`include "dump_file_agent.sv"
class sample_agent;
  dump_file_agent file_dumper;

  function new(dump_file_agent file_inst);
    this.file_dumper = file_inst;
  endfunction

  virtual function void output_to_file();
  endfunction

  virtual task data_monitor();
  endtask

endclass
`endif
