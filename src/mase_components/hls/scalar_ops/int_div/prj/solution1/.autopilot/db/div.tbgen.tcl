set moduleName div
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {div}
set C_modelType { void 0 }
set C_modelArgList {
	{ data_in_0 int 32 regular {fifo 0 volatile }  }
	{ data_in_1 int 32 regular {fifo 0 volatile }  }
	{ data_out_0 int 16 regular {fifo 1 volatile }  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "data_in_0", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "data_in_1", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "data_out_0", "interface" : "fifo", "bitwidth" : 16, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 15
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ data_in_0_dout sc_in sc_lv 32 signal 0 } 
	{ data_in_0_empty_n sc_in sc_logic 1 signal 0 } 
	{ data_in_0_read sc_out sc_logic 1 signal 0 } 
	{ data_in_1_dout sc_in sc_lv 32 signal 1 } 
	{ data_in_1_empty_n sc_in sc_logic 1 signal 1 } 
	{ data_in_1_read sc_out sc_logic 1 signal 1 } 
	{ data_out_0_din sc_out sc_lv 16 signal 2 } 
	{ data_out_0_full_n sc_in sc_logic 1 signal 2 } 
	{ data_out_0_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "data_in_0_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "data_in_0", "role": "dout" }} , 
 	{ "name": "data_in_0_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_in_0", "role": "empty_n" }} , 
 	{ "name": "data_in_0_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_in_0", "role": "read" }} , 
 	{ "name": "data_in_1_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "data_in_1", "role": "dout" }} , 
 	{ "name": "data_in_1_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_in_1", "role": "empty_n" }} , 
 	{ "name": "data_in_1_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_in_1", "role": "read" }} , 
 	{ "name": "data_out_0_din", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "data_out_0", "role": "din" }} , 
 	{ "name": "data_out_0_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_out_0", "role": "full_n" }} , 
 	{ "name": "data_out_0_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_out_0", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "div",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "35", "EstimateLatencyMin" : "35", "EstimateLatencyMax" : "35",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "1",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data_in_0", "Type" : "Fifo", "Direction" : "I"},
			{"Name" : "data_in_1", "Type" : "Fifo", "Direction" : "I"},
			{"Name" : "data_out_0", "Type" : "Fifo", "Direction" : "O"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sdiv_32ns_32ns_16_36_1_U1", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	div {
		data_in_0 {Type I LastRead 0 FirstWrite -1}
		data_in_1 {Type I LastRead 0 FirstWrite -1}
		data_out_0 {Type O LastRead 35 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "35", "Max" : "35"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	data_in_0 { ap_fifo {  { data_in_0_dout fifo_port_we 0 32 }  { data_in_0_empty_n fifo_status 0 1 }  { data_in_0_read fifo_data 1 1 } } }
	data_in_1 { ap_fifo {  { data_in_1_dout fifo_port_we 0 32 }  { data_in_1_empty_n fifo_status 0 1 }  { data_in_1_read fifo_data 1 1 } } }
	data_out_0 { ap_fifo {  { data_out_0_din fifo_port_we 1 16 }  { data_out_0_full_n fifo_status 0 1 }  { data_out_0_write fifo_data 1 1 } } }
}

set maxi_interface_dict [dict create]

# RTL port scheduling information:
set fifoSchedulingInfoList { 
	data_in_0 { fifo_read 2 has_conditional }
	data_in_1 { fifo_read 2 has_conditional }
	data_out_0 { fifo_write 1 has_conditional }
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
