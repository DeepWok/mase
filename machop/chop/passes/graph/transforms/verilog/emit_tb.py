import math, time, os, logging, torch, glob, shutil

from chop.passes.graph.utils import vf, v2p, init_project
from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer_for_hw
from .emit_tb_data_in import emit_data_in_tb_sv, emit_data_in_tb_dat
from .emit_tb_data_out import emit_data_out_tb_sv, emit_data_out_tb_dat
from .emit_tb_testbench import emit_top_tb

logger = logging.getLogger(__name__)


def emit_tb_verilog(graph, trans_num=1, project_dir="top"):
    sim_dir = os.path.join(project_dir, "hardware", "sim")
    tv_dir = os.path.join(sim_dir, "tv")
    if not os.path.exists(tv_dir):
        os.mkdir(tv_dir)
    v_dir = os.path.join(sim_dir, "verilog")
    if not os.path.exists(v_dir):
        os.mkdir(v_dir)

    # TODO : need to emit all the inputs
    v_in_param = (
        graph.nodes_in[0].meta["mase"].parameters["hardware"]["verilog_parameters"]
    )
    w_in_param = graph.nodes_in[0].meta["mase"].parameters["common"]["args"]
    in_width = w_in_param["data_in_0"]["precision"][0]
    in_size = v_in_param["IN_0_SIZE"]
    data_width = in_width * in_size
    # TODO : need to check
    addr_width = 1
    depth = 1
    load_path = os.path.join(tv_dir, f"sw_data_in.dat")
    out_file = os.path.join(v_dir, f"top_data_in_fifo.sv")
    emit_data_in_tb_sv(data_width, load_path, out_file)

    v_out_param = (
        graph.nodes_out[0].meta["mase"].parameters["hardware"]["verilog_parameters"]
    )
    w_out_param = graph.nodes_in[0].meta["mase"].parameters["common"]["results"]
    out_width = w_out_param["data_out_0"]["precision"][0]
    out_size = v_out_param["OUT_0_SIZE"]
    data_width = out_width * out_size
    # TODO : need to check
    addr_width = 1
    depth = 1
    load_path = os.path.join(tv_dir, f"sw_data_out.dat")
    store_path = os.path.join(tv_dir, f"hw_data_out.dat")
    out_file = os.path.join(v_dir, f"top_data_out_fifo.sv")
    emit_data_out_tb_sv(data_width, load_path, store_path, out_file)

    out_file = os.path.join(v_dir, f"top_tb.sv")
    in_trans_num = v_in_param["IN_0_DEPTH"] * trans_num
    out_trans_num = trans_num
    emit_top_tb(
        tv_dir,
        "top",
        out_file,
        in_width,
        in_size,
        out_width,
        out_size,
        in_trans_num,
        out_trans_num,
    )

    out_file = os.path.join(v_dir, f"fifo_para.v")
    with open(out_file, "w", encoding="utf-8") as outf:
        outf.write("// initial a empty file")

    # Copy testbench components
    dut_dir = os.path.join(project_dir, "hardware", "rtl")
    for svfile in glob.glob(os.path.join(dut_dir, "*.sv")):
        shutil.copy(svfile, v_dir)


def emit_tb_dat(graph, trans_num=1, project_dir="top", test_inputs=None):
    """
    Emit the test vectors in dat files for simulation
    """

    sim_dir = os.path.join(project_dir, "hardware", "sim")
    tv_dir = os.path.join(sim_dir, "tv")
    if not os.path.exists(tv_dir):
        os.mkdir(tv_dir)

    in_type = (
        graph.nodes_in[0].meta["mase"].parameters["common"]["args"]["data_in_0"]["type"]
    )
    in_width = (
        graph.nodes_in[0]
        .meta["mase"]
        .parameters["common"]["args"]["data_in_0"]["precision"][0]
    )
    in_frac_width = (
        graph.nodes_in[0]
        .meta["mase"]
        .parameters["common"]["args"]["data_in_0"]["precision"][1]
    )

    sw_data_out = [graph.model(trans) for trans in test_inputs]

    out_type = (
        graph.nodes_out[0]
        .meta["mase"]
        .parameters["common"]["results"]["data_out_0"]["type"]
    )
    out_width = (
        graph.nodes_out[0]
        .meta["mase"]
        .parameters["common"]["results"]["data_out_0"]["precision"][0]
    )
    out_frac_width = (
        graph.nodes_out[0]
        .meta["mase"]
        .parameters["common"]["results"]["data_out_0"]["precision"][1]
    )

    # TODO: Make out_type as input to support casting to any type
    hw_data_out = [
        integer_quantizer_for_hw(trans, width=out_width, frac_width=out_frac_width)
        .squeeze(0)
        .to(torch.int)
        for trans in sw_data_out
    ]

    # TODO: for now
    for i, trans in enumerate(test_inputs):
        test_inputs[i] = torch.flatten(trans).tolist()
    for i, trans in enumerate(hw_data_out):
        hw_data_out[i] = torch.flatten(trans).tolist()

    load_path = os.path.join(tv_dir, "sw_data_in.dat")
    emit_data_in_tb_dat(graph.nodes_in[0], test_inputs, load_path)

    load_path = os.path.join(tv_dir, "sw_data_out.dat")
    emit_data_out_tb_dat(graph.nodes_out[0], hw_data_out, load_path)


def emit_tb_tcl(graph, project_dir="top"):
    """
    Emit Vivado tcl files for simulation
    """
    sim_dir = os.path.join(project_dir, "hardware", "sim")
    prj_dir = os.path.join(sim_dir, "prj")
    if not os.path.exists(prj_dir):
        os.mkdir(prj_dir)

    out_file = os.path.join(prj_dir, "proj.tcl")
    buff = f"""
#log_wave -r /
run all
quit
"""
    with open(out_file, "w", encoding="utf-8") as outf:
        outf.write(buff)

    rtl_dir = os.path.join(prj_dir, "..", "..", "rtl")
    v_dir = os.path.join(prj_dir, "..", "verilog")

    buff = ""
    for file_dir in [rtl_dir, v_dir]:
        for file in glob.glob(os.path.join(file_dir, "*.sv")) + glob.glob(
            os.path.join(file_dir, "*.v")
        ):
            buff += f"""sv work "{file}"
"""
            for file in glob.glob(os.path.join(file_dir, "*.vhd")):
                buff += f"""vhdl work "{file}"
"""

    # Add HLS tcls
    hls_dir = os.path.join(prj_dir, "..", "..", "hls")
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["hardware"]["is_implicit"]:
            continue
        if "HLS" not in node.meta["mase"].parameters["hardware"]["toolchain"]:
            continue
        node_name = vf(node.name)
        syn_dir = os.path.join(
            hls_dir, node_name, node_name, "solution1", "syn", "verilog"
        )
        assert os.path.exists(syn_dir), f"Cannot find path: {syn_dir}"
        for file in glob.glob(os.path.join(syn_dir, "*.v")):
            buff += f"""sv work "{file}"
"""
            for file in glob.glob(os.path.join(syn_dir, "*.tcl")):
                buff += f"""source "{file}"
"""

    out_file = os.path.join(prj_dir, "proj.prj")
    with open(out_file, "w", encoding="utf-8") as outf:
        outf.write(buff)


def emit_verilog_tb_transform_pass(graph, pass_args={}):
    """
    Emit test bench and related files for simulation
    * project_dir : the directory of the project for cosimulation
    * trans_num : the transaction count of cosimulation
    * test_inputs : test vectors of inputs for cosimulation
    """
    logger.info("Emitting test bench...")
    project_dir = (
        pass_args["project_dir"] if "project_dir" in pass_args.keys() else "top"
    )
    trans_num = pass_args["trans_num"] if "trans_num" in pass_args.keys() else 1
    test_inputs = (
        pass_args["test_inputs"] if "test_inputs" in pass_args.keys() else None
    )
    assert len(test_inputs) == trans_num

    init_project(project_dir)
    emit_tb_verilog(graph, trans_num=trans_num, project_dir=project_dir)
    emit_tb_dat(
        graph, trans_num=trans_num, project_dir=project_dir, test_inputs=test_inputs
    )
    emit_tb_tcl(graph, project_dir=project_dir)
    return graph
