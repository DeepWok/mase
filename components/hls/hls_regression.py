#! /usr/bin/env python3
# ---------------------------------------
# This script runs the regression model for hls
# ---------------------------------------

from argparse import ArgumentParser
import os

from regression_gen import (
    int_linear2d_dse,
    int_rmsnorm_dse,
    int_rope_dse,
    int_softmax_dse,
    int_layernorm_dse,
    int_mult_dse,
    int_add_dse,
    int_relu_dse,
    int_silu_dse,
    int_transpose_dse,
    int_matmul_dse,
    fork_dse,
    buffer_dse,
    bfp_add_dse,
    bfp_mult_dse,
    bfp_linear2d_dse,
)


def run(args):
    op = args.op
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    mode = args.mode
    top = args.dir
    if op == "int_linear2d":
        int_linear2d_dse(mode=mode, top=top)
    elif op == "int_softmax":
        int_softmax_dse(mode=mode, top=top)
    elif op == "int_rmsnorm":
        int_rmsnorm_dse(mode=mode, top=top)
    elif op == "int_rope":
        int_rope_dse(mode=mode, top=top)
    elif op == "int_layernorm":
        int_layernorm_dse(mode=mode, top=top)
    elif op == "int_mult":
        int_mult_dse(mode=mode, top=top)
    elif op == "int_add":
        int_add_dse(mode=mode, top=top)
    elif op == "int_relu":
        int_relu_dse(mode=mode, top=top)
    elif op == "int_silu":
        int_silu_dse(mode=mode, top=top)
    elif op == "int_transpose":
        int_transpose_dse(mode=mode, top=top)
    elif op == "int_matmul":
        int_matmul_dse(mode=mode, top=top)
    elif op == "fork":
        fork_dse(mode=mode, top=top)
    elif op == "buffer":
        buffer_dse(mode=mode, top=top)
    elif op == "bfp_add":
        bfp_add_dse(mode=mode, top=top)
    elif op == "bfp_mult":
        bfp_mult_dse(mode=mode, top=top)
    elif op == "bfp_linear2d":
        bfp_linear2d_dse(mode=mode, top=top)
    else:
        assert False, f"Unsupported op = {op}"


# ---------- main function --------------
def main():
    USAGE = """Usage:
mase_hls  ...
"""

    parser = ArgumentParser(usage=USAGE)
    parser.add_argument(
        "--op",
        dest="op",
        default=None,
        help="Op name to explore",
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        default=os.path.join(os.getcwd(), "dse"),
        help="Directory to store files",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        default=None,
        help="Mode to run: codegen, synth, report, all",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
