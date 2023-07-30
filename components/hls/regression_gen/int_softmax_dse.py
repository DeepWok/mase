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
from hls.int_arith import int_softmax_gen
from hls import HLSWriter


def int_softmax_dse(mode=None, top=None, threads=16):
    assert mode in DSE_MODES, f"Unknown mode {mode}"

    # Small size for debugging only
    # x_widths = [8]
    # x_frac_widths = [1]
    # x_rows = [5]
    # x_cols = [7]

    x_widths = [1, 2, 3, 4, 5, 6, 7, 8]
    x_frac_widths = [1]
    x_rows = [1, 2, 3, 4, 5, 6, 7, 8]
    x_cols = [1, 2, 3, 4, 5, 6, 7, 8]
    # Ignored to reduce complexity
    x_row_depth = 8
    x_col_depth = 8

    data_points = []
    data_points.append(
        [
            "x_width",
            "x_frac_width",
            "x_row",
            "x_col",
            "x_row_depth",
            "x_col_depth",
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

    size = len(x_widths) * len(x_frac_widths) * len(x_rows) * len(x_cols)
    print("Exploring softmax. Design points = {}".format(size))

    i = 0
    commands = [[] for i in range(0, threads)]
    for x_row in x_rows:
        for x_col in x_cols:
            for x_width in x_widths:
                for x_frac_width in x_frac_widths:
                    print(f"Running design {i}/{size}")

                    file_name = (
                        f"x{i}_int_softmax_{x_row}_{x_col}_{x_width}_{x_frac_width}"
                    )
                    tcl_path = os.path.join(top, f"{file_name}.tcl")
                    file_path = os.path.join(top, f"{file_name}.cpp")
                    if mode in ["codegen", "all"]:
                        writer = HLSWriter()
                        writer = int_softmax_gen(
                            writer,
                            x_width=x_width,
                            x_frac_width=x_frac_width,
                            x_row=x_row,
                            x_col=x_col,
                            x_row_depth=x_row_depth,
                            x_col_depth=x_col_depth,
                        )
                        writer.emit(file_path)
                        os.system("clang-format -i {}".format(file_path))
                        top_name = f"int_softmax_{writer.op_id-1}"
                        tcl_buff = get_tcl_buff(
                            project=file_name, top=top_name, cpp=f"{file_name}.cpp"
                        )
                        with open(tcl_path, "w", encoding="utf-8") as outf:
                            outf.write(tcl_buff)
                        commands[i % threads].append(
                            f'echo "{i}/{size}"; vitis_hls {file_name}.tcl'
                        )

                    if mode in ["synth", "all"]:
                        os.system(f"cd {top}; vitis_hls {file_name}.tcl")

                    if mode in ["report", "all"]:
                        top_name = "int_softmax_0"
                        hr = get_hls_results(
                            project=os.path.join(top, file_name),
                            top=top_name,
                        )
                        data_points.append(
                            [
                                x_width,
                                x_frac_width,
                                x_row,
                                x_col,
                                x_row_depth,
                                x_col_depth,
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
        bash_gen(commands, top, "int_softmax")

    if mode in ["report", "all"]:
        # Export regression model data points to csv
        csv_gen(data_points, top, "int_softmax")
