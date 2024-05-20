# mmult.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air.compiler.util

from air.mlir.dialects import func, linalg, tensor, arith
from air.mlir.ir import *
import air.mlir.passmanager

import sys

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
Tiling_L1_m = int(sys.argv[4])
Tiling_L1_n = int(sys.argv[5])
Tiling_L1_k = int(sys.argv[6])
Tiling_L2_m = int(sys.argv[7])
Tiling_L2_n = int(sys.argv[8])
Tiling_L2_k = int(sys.argv[9])


def matmul_on_tensors(m, n, k, dtype):
    module = Module.create()
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            RankedTensorType.get((m, k), dtype), RankedTensorType.get((k, n), dtype)
        )
        def matmul(lhs, rhs):
            out = tensor.EmptyOp([m, n], dtype)
            zero = arith.ConstantOp(dtype, 0.0)
            zero_fill = linalg.fill(zero, outs=[out])
            return linalg.matmul(lhs, rhs, outs=[zero_fill])

    return module


with air.mlir.ir.Context(), Location.unknown():
    air_module = matmul_on_tensors(M, N, K, BF16Type.get())

    # convert linalg on tensors to linalg on memrefs
    pm = air.mlir.passmanager.PassManager.parse(
        air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE
    )
    pm.run(air_module)

    args = sys.argv[1:]
    if len(args) and args[0] == "-dump-linalg":
        print(air_module)
        exit(0)

    # tile and map to air
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "buffer-results-to-out-params",
                "air-linalg-codegen{l2-tile-size="
                + str(Tiling_L2_m)
                + ","
                + str(Tiling_L2_n)
                + ","
                + str(Tiling_L2_k)
                + " l2-promote=true l1-tile-size="
                + str(Tiling_L1_m)
                + ","
                + str(Tiling_L1_n)
                + ","
                + str(Tiling_L1_k)
                + " l1-promote=true}",
                "canonicalize",
                "cse",
                "air-par-to-herd{depth=1}",
                "air-copy-to-dma",
                "air-par-to-launch{has-air-segment=true}",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    print("\nAIR Dialect Module\n")
    print(air_module)

    # generate dependency information for runner
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-dependency",
                "air-dependency-schedule-opt",
                "air-specialize-dma-broadcast",
                "air-dma-to-channel",
                "canonicalize",
                "cse",
                "air-dependency-canonicalize",
                "air-dependency-parse-graph{output-dir=dot_graphs/}",
                "canonicalize",
                "cse",
                "air-place-herds{num-rows=2 num-cols=2 row-anchor=0 col-anchor=0}",
                "air-label-scf-for-to-ping-pong",
                "air-ping-pong-transform",
            ]
        )
        + ")"
    )
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    print("\nAIR Dialect Module (async)\n")
    print(air_module)

    arch = {
        "clock": 1000000000,
        "cores": 1,
        "datatypes": [
            {"bytes": 1, "name": "i8"},
            {"bytes": 2, "name": "bf16"},
            {"bytes": 4, "name": "i32"},
        ],
        "devicename": "testdevice",
        "kernels": {
            "linalg.copy": {
                "datatypes": {
                    "i8": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                    "bf16": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                    "i32": {"ops_per_core_per_cycle": 16, "efficiency": 1},
                },
                "name": "linalg.copy",
            },
            "linalg.fill": {
                "datatypes": {
                    "i8": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                    "bf16": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                    "i32": {"ops_per_core_per_cycle": 16, "efficiency": 1},
                },
                "name": "linalg.fill",
            },
            "linalg.generic": {
                "datatypes": {
                    "i8": {"ops_per_core_per_cycle": 1, "efficiency": 1},
                    "bf16": {"ops_per_core_per_cycle": 1, "efficiency": 1},
                    "i32": {"ops_per_core_per_cycle": 1, "efficiency": 1},
                },
                "name": "linalg.generic",
            },
            "linalg.matmul": {
                "datatypes": {
                    "i8": {"macs_per_core_per_cycle": 256, "efficiency": 1},
                    "bf16": {"macs_per_core_per_cycle": 128, "efficiency": 1},
                    "i32": {"macs_per_core_per_cycle": 32, "efficiency": 1},
                },
                "name": "linalg.matmul",
            },
        },
        "dus": {
            "count": [4, 4],
            "memory": {"memory_space": "L2", "bytes": 524288},
            "ports": {
                "outbound": {"count": 6, "bytes_per_second": 4000000000},
                "inbound": {"count": 6, "bytes_per_second": 4000000000},
            },
            "tiles": {
                "count": [1, 4],
                "memory": {"memory_space": "L1", "bytes": 65536},
                "ports": {
                    "outbound": {"count": 2, "bytes_per_second": 4000000000},
                    "inbound": {"count": 2, "bytes_per_second": 4000000000},
                },
            },
        },
        "noc": {
            "outbound": {"count": 4, "bytes_per_second": 4000000000},
            "inbound": {"count": 4, "bytes_per_second": 4000000000},
        },
    }

runner = air.compiler.util.Runner(arch, "trace.out", "core")
trace = runner.run(air_module, "matmul")
