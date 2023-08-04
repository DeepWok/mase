# TODO: Temporary working solution
import sys, os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
    )
)

from hls.regression_gen.utils import (
    DSE_MODES,
    get_tcl_buff,
    get_hls_results,
    bash_gen,
    csv_gen,
)
from hls.bfp_arith import bfp_linear2d_gen
from hls import HLSWriter


def bfp_linear2d_dse(mode=None, top=None, threads=32):
    assert mode in DSE_MODES, f"Unknown mode {mode}"

    # Small size for debugging only
    # x_exp_widths = [4]
    # x_man_widths = [4]
    # x_rows = [2]
    # x_cols = [3]
    # w_exp_widths = [2]
    # w_man_widths = [2]
    # w_rows = [4]

    x_exp_widths = [2, 4, 6, 8]
    x_man_widths = [2, 4, 6]
    x_rows = [1, 2, 3, 4]
    x_cols = [1, 2, 3, 4]
    w_exp_widths = [2, 4, 6, 8]
    w_man_widths = [2, 4, 6]
    w_rows = [1, 2, 3, 4]

    # Ignored to reduce complexity
    w_row_depths = [2]
    w_col_depths = [2]
    x_row_depths = [2]
    x_col_depths = [2]
    b_exp_widths = [8]
    b_man_widths = [4]

    size = (
        len(x_exp_widths)
        * len(x_man_widths)
        * len(x_rows)
        * len(x_cols)
        * len(w_exp_widths)
        * len(w_man_widths)
        * len(w_rows)
    )

    # Ignored to reduce complexity
    print("Exploring linear2d. Design points = {}".format(size))
    w_row_depth = 8
    w_col_depth = 8
    x_row_depth = 8
    x_col_depth = 8
    b_exp_width = 0
    b_man_width = 0

    i = 0
    commands = [[] for i in range(0, threads)]
    data_points = []
    data_points.append(
        [
            "x_exp_width",
            "x_man_width",
            "x_row",
            "x_col",
            "x_row_depth",
            "x_col_depth",
            "w_exp_width",
            "w_man_width",
            "w_row",
            "w_col",
            "w_row_depth",
            "w_col_depth",
            "latency_min",
            "latency_max",
            "clock_period",
            "bram",
            "dsp",
            "ff",
            "lut",
            "uram",
        ]
    )
    for x_row in x_rows:
        w_col = x_row
        for x_col in x_cols:
            for x_exp_width in x_exp_widths:
                for x_man_width in x_man_widths:
                    for w_row in w_rows:
                        for w_exp_width in w_exp_widths:
                            for w_man_width in w_man_widths:
                                print(f"Running design {i}/{size}")

                                file_name = f"x{i}_bfp_linear2d_{x_row}_{x_col}_{x_exp_width}_{x_man_width}_{w_row}_{w_col}_{w_exp_width}_{w_man_width}"
                                tcl_path = os.path.join(top, f"{file_name}.tcl")
                                file_path = os.path.join(top, f"{file_name}.cpp")
                                if mode in ["codegen", "all"]:
                                    writer = HLSWriter()
                                    writer = bfp_linear2d_gen(
                                        writer,
                                        x_exp_width=x_exp_width,
                                        x_man_width=x_man_width,
                                        x_row=x_row,
                                        x_col=x_col,
                                        x_row_depth=x_row_depth,
                                        x_col_depth=x_col_depth,
                                        w_exp_width=w_exp_width,
                                        w_man_width=w_man_width,
                                        w_row=w_row,
                                        w_col=w_col,
                                        w_row_depth=w_row_depth,
                                        w_col_depth=w_col_depth,
                                        b_exp_width=b_exp_width,
                                        b_man_width=b_man_width,
                                    )
                                    writer.emit(file_path)
                                    os.system("clang-format -i {}".format(file_path))
                                    top_name = f"bfp_linear2d_{writer.op_id-1}"
                                    tcl_buff = get_tcl_buff(
                                        project=file_name,
                                        top=top_name,
                                        cpp=f"{file_name}.cpp",
                                    )
                                    with open(tcl_path, "w", encoding="utf-8") as outf:
                                        outf.write(tcl_buff)
                                    commands[i % threads].append(
                                        f'echo "{i}/{size}"; vitis_hls {file_name}.tcl'
                                    )

                                if mode in ["synth", "all"]:
                                    os.system(f"cd {top}; vitis_hls {file_name}.tcl")

                                if mode in ["report", "all"]:
                                    top_name = "bfp_linear2d_1"
                                    hr = get_hls_results(
                                        project=os.path.join(top, file_name),
                                        top=top_name,
                                    )
                                    data_points.append(
                                        [
                                            x_exp_width,
                                            x_man_width,
                                            x_row,
                                            x_col,
                                            x_row_depth,
                                            x_col_depth,
                                            w_exp_width,
                                            w_man_width,
                                            w_row,
                                            w_col,
                                            w_row_depth,
                                            w_col_depth,
                                            hr.latency_min,
                                            hr.latency_max,
                                            hr.clock_period,
                                            hr.bram,
                                            hr.dsp,
                                            hr.ff,
                                            hr.lut,
                                            hr.uram,
                                        ]
                                    )

                                i += 1

    if mode in ["codegen", "all"]:
        # Generate bash script for running HLS in parallel
        bash_gen(commands, top, "bfp_linear2d")

    if mode in ["report", "all"]:
        # Export regression model data points to csv
        csv_gen(data_points, top, "bfp_linear2d")
