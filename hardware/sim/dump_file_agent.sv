// base class for all dataflow file dumpers
`ifndef DUMP_FILE_AGENT__SV
`define DUMP_FILE_AGENT__SV
class dump_file_agent;
  string file_name;

  //initial function
  function new(string name);
    this.file_name = name;
  endfunction

  virtual function integer get_file_handle();
  endfunction

  virtual function void open_file();
  endfunction

  virtual function void dump_1_col(logic [31:0] arr[$]);
  endfunction

  virtual function void dump_2_col(logic [31:0] arr1[$], logic [1:0] arr2[$]);
  endfunction

  virtual function void dump_1_line(logic [31:0] arr1[$], logic [31:0] arr2[$]);
  endfunction

  virtual function void finish_dump();
  endfunction

endclass
`endif
