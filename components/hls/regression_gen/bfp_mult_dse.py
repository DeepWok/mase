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
from hls.bfp_arith import bfp_mult_gen
from hls import HLSWriter


def bfp_mult_dse(mode=None, top=None, threads=16):
    assert mode in DSE_MODES, f"Unknown mode {mode}"

    # Small size for debugging only
    # x_exp_widths = [1]
    # x_man_widths = [1]
    # x_rows = [2]
    # x_cols = [3]
    # w_exp_widths = [1]
    # w_man_widths = [1]

    x_exp_widths = [2, 4, 6, 8]
    x_man_widths = [2, 4, 6, 8]
    w_exp_widths = [2, 4, 6, 8]
    w_man_widths = [2, 4, 6, 8]
    x_rows = [1, 2, 4, 6, 8]
    x_cols = [1, 2, 4, 6, 8]

    # Ignored to reduce complexity
    x_row_depths = [8]
    x_col_depths = [8]

    data_pobfps = []
    data_pobfps.append(
        [
            "x_exp_width",
            "x_man_width",
            "x_row",
            "x_col",
            "x_row_depth",
            "x_col_depth",
            "w_exp_width",
            "w_man_width",
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

    size = (
        len(x_exp_widths)
        * len(x_man_widths)
        * len(x_rows)
        * len(x_cols)
        * len(w_exp_widths)
        * len(w_man_widths)
    )
    print("Exploring mult. Design points = {}".format(size))

    i = 0
    commands = [[] for i in range(0, threads)]
    for x_row in x_rows:
        for x_col in x_cols:
            for x_exp_width in x_exp_widths:
                for x_man_width in x_man_widths:
                    for w_exp_width in w_exp_widths:
                        for w_man_width in w_man_widths:
                            print(f"Running design {i}/{size}")
                            # Ignored to reduce complexity
                            w_exp_width = x_exp_width
                            w_man_width = x_man_width
                            x_row_depth = 8
                            x_col_depth = 8

                            file_name = f"x{i}_bfp_mult_{x_row}_{x_col}_{x_exp_width}_{x_man_width}_{w_exp_width}_{w_man_width}"
                            tcl_path = os.path.join(top, f"{file_name}.tcl")
                            file_path = os.path.join(top, f"{file_name}.cpp")
                            if mode in ["codegen", "all"]:
                                writer = HLSWriter()
                                writer = bfp_mult_gen(
                                    writer,
                                    x_exp_width=x_exp_width,
                                    x_man_width=x_man_width,
                                    x_row=x_row,
                                    x_col=x_col,
                                    x_row_depth=x_row_depth,
                                    x_col_depth=x_col_depth,
                                    w_exp_width=w_exp_width,
                                    w_man_width=w_man_width,
                                )
                                writer.emit(file_path)
                                os.system("clang-format -i {}".format(file_path))
                                top_name = f"bfp_mult_{writer.op_id-1}"
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
                                top_name = "bfp_mult_2"
                                hr = get_hls_results(
                                    project=os.path.join(top, file_name),
                                    top=top_name,
                                )
                                if hr is None:
                                    continue
                                data_pobfps.append(
                                    [
                                        x_exp_width,
                                        x_man_width,
                                        x_row,
                                        x_col,
                                        x_row_depth,
                                        x_col_depth,
                                        w_exp_width,
                                        w_man_width,
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
        bash_gen(commands, top, "bfp_mult")

    if mode in ["report", "all"]:
        # Export regression model data pobfps to csv
        csv_gen(data_pobfps, top, "bfp_mult")
