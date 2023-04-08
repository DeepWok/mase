
`include "sample_manager.sv"
`include "csv_file_dump.sv"
`include "df_fifo_monitor.sv"
`include "df_process_monitor.sv"
`include "nodf_module_monitor.sv"
`timescale 1ns / 1ps

// top module for dataflow related monitors
module dataflow_monitor (
    input logic clock,
    input logic reset,
    input logic finish
);

  nodf_module_intf module_intf_1 (
      clock,
      reset
  );

  assign module_intf_1.ap_continue = 1'b1;
  assign module_intf_1.finish = finish;
  csv_file_dump mstatus_csv_dumper_1;
  nodf_module_monitor module_monitor_1;

  sample_manager sample_manager_inst;

  initial begin
    sample_manager_inst = new;



    mstatus_csv_dumper_1 = new("./module_status1.csv");
    module_monitor_1 = new(module_intf_1, mstatus_csv_dumper_1);

    sample_manager_inst.add_one_monitor(module_monitor_1);

    fork
      sample_manager_inst.start_monitor();
      last_transaction_done;
    join
    disable fork;

    sample_manager_inst.start_dump();
  end

  task last_transaction_done();
    wait (reset == 0);
    while (1) begin
      if (finish == 1'b1) break;
      else @(posedge clock);
    end
  endtask

endmodule
