import numpy as np
import functools
import itertools
import math
import copy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv
import toml
import os


def get_factors(n):
    return np.sort(
        list(
            set(
                functools.reduce(
                    list.__add__,
                    ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
                )
            )
        )
    )


class Variable:
    def __init__(self, row_dim, col_dim):
        self.row_dim = row_dim
        self.col_dim = col_dim

        self.row = 1
        self.col = 1

        self.row_depth = self.row_dim // self.row
        self.col_depth = self.col_dim // self.col

    def get_row_feasible(self):
        return get_factors(self.row_dim)

    def update_row(self, x):
        self.row = x
        self.row_depth = self.row_dim // self.row

    def update_col(self, x):
        self.col = x
        self.col_depth = self.col_dim // self.col


class AttentionBlockI:
    def __init__(self, tunable, target_len, num_head, embed_dim, head_dim):
        self.target_len = target_len
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.variables = {
            "x": Variable(embed_dim, target_len),
            "layernorm0": Variable(embed_dim, target_len),
            "w": Variable(head_dim, embed_dim),
            "Q": Variable(head_dim, target_len),
            "K": Variable(head_dim, target_len),
            "V": Variable(head_dim, target_len),
            "K^T": Variable(target_len, head_dim),
            "A": Variable(target_len, target_len),
            "A^bar": Variable(target_len, target_len),
            "B": Variable(head_dim, target_len),
            "B_concat": Variable(embed_dim, target_len),
            "W_0": Variable(embed_dim, embed_dim),
            "B_multi": Variable(embed_dim, target_len),
            "add1": Variable(embed_dim, target_len),
            "layernorm1": Variable(embed_dim, target_len),
            "W_1": Variable(int(4 * embed_dim), embed_dim),
            "linear1": Variable(int(4 * embed_dim), embed_dim),
            "relu": Variable(int(4 * embed_dim), embed_dim),
            "W_2": Variable(embed_dim, int(4 * embed_dim)),
            "linear2": Variable(embed_dim, target_len),
            "add2": Variable(embed_dim, target_len),
        }

        self.tunable = tunable
        self.mm = {
            "Q": ("w", "layernorm0"),
            "K": ("w", "layernorm0"),
            "V": ("w", "layernorm0"),
            "A": ("K^T", "Q"),
            "B": ("V", "A^bar"),
            "B_multi": ("W_0", "B_concat"),
            "linear1": ("W_1", "layernorm1"),
            "linear2": ("W_2", "relu"),
        }

    def get_row_feasible(self, var_name):
        feasible = {self.variables[n].get_row_feasible() for n in self.tunable}
        return feasible

    def update(self, config, input_II=1):
        for i, n in enumerate(self.variables.keys()):
            if n in config.keys():
                row = config[n]
            elif n == "K^T":
                row = self.variables["Q"].col
            elif n == "B":
                row = self.variables["V"].row
            else:
                row = list(self.variables.values())[i - 1].row
            self.variables[n].update_row(row)

        for n1, n2 in self.mm.values():
            self.variables[n1].update_col(self.variables[n2].row)

        def _MM_II(n1, n2):
            return math.ceil(
                max(n2.output_II, n1.row_depth) * n2.row_depth / n1.row_depth
            )

        for i, n in enumerate(self.variables.keys()):
            if n == "x":
                self.variables["x"].output_II = input_II
            elif n in self.mm.keys():
                n1, n2 = self.mm[n]
                self.variables[n].output_II = _MM_II(
                    self.variables[n1], self.variables[n2]
                )
            elif n == "B_concat":
                self.variables["B_concat"].output_II = (
                    self.variables["B"].output_II / self.num_head
                )
            elif n in ["w", "W_0", "W_1", "W_2"]:
                continue
            else:
                self.variables[n].output_II = list(self.variables.values())[
                    i - 1
                ].output_II

    def get_latency(self):
        last = list(self.variables.values())[-1]
        effective_rate = last.row * last.col / last.output_II
        return last.row_dim * last.col_dim / effective_rate

    def get_resource(self, rsc_predictors, mixed_precision, breakdown):
        if breakdown:
            rsc = {}
            rsc["softmax"] = (
                rsc_predictors["softmax"].predict([[self.variables["A^bar"].row]])[0]
                * self.num_head
            )
            rsc["layernorm0"] = rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm0"].row]]
            )[0]
            rsc["layernorm1"] = rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm1"].row]]
            )[0]
            for k, (n1, n2) in self.mm.items():
                if mixed_precision:
                    pass
                else:
                    mm_rsc = rsc_predictors["matmul"].predict(
                        [
                            [
                                self.variables[n2].row,
                                self.variables[n1].row,
                                int(self.variables[n2].row * self.variables[n1].row),
                            ]
                        ]
                    )[0]
                if k in ["Q", "K", "V", "A", "B"]:
                    mm_rsc *= self.num_head
                rsc[k] = mm_rsc
        else:
            rsc = 0
            rsc += (
                rsc_predictors["softmax"].predict([[self.variables["A^bar"].row]])[0]
                * self.num_head
            )
            rsc += rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm0"].row]]
            )[0]
            rsc += rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm1"].row]]
            )[0]
            for k, (n1, n2) in self.mm.items():
                if mixed_precision:
                    i = self.variables[n2].row
                    j = self.variables[n1].row
                    m = self.variables[k].data_in_width
                    n = self.variables[k].weight_width
                    features = [
                        [
                            i,
                            j,
                            m,
                            n,
                            int(i * j),
                            int(i * m),
                            int(i * n),
                            int(j * m),
                            int(j * n),
                            int(m * n),
                            int(i * j * m),
                            int(i * j * n),
                            int(i * m * n),
                            int(j * m * n),
                            int(i * j * m * n),
                        ]
                    ]
                    mm_rsc = rsc_predictors["matmul"].predict(features)[0]
                else:
                    mm_rsc = rsc_predictors["matmul"].predict(
                        [
                            [
                                self.variables[n2].row,
                                self.variables[n1].row,
                                int(self.variables[n2].row * self.variables[n1].row),
                            ]
                        ]
                    )[0]
                if k in ["Q", "K", "V", "A", "B"]:
                    mm_rsc *= self.num_head
                rsc += mm_rsc
        return rsc


class AttentionBlockII(AttentionBlockI):
    def __init__(self, tunable, target_len, num_head, embed_dim, head_dim):
        self.target_len = target_len
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.variables = {
            "x": Variable(embed_dim, target_len),
            "w": Variable(head_dim, embed_dim),
            "Q": Variable(head_dim, target_len),
            "K": Variable(head_dim, target_len),
            "V": Variable(head_dim, target_len),
            "K^T": Variable(target_len, head_dim),
            "A": Variable(target_len, target_len),
            "A^bar": Variable(target_len, target_len),
            "B": Variable(head_dim, target_len),
            "B_concat": Variable(embed_dim, target_len),
            "W_0": Variable(embed_dim, embed_dim),
            "B_multi": Variable(embed_dim, target_len),
            "add1": Variable(embed_dim, target_len),
            "layernorm1": Variable(embed_dim, target_len),
            "W_1": Variable(int(4 * embed_dim), embed_dim),
            "linear1": Variable(int(4 * embed_dim), embed_dim),
            "relu": Variable(int(4 * embed_dim), embed_dim),
            "W_2": Variable(embed_dim, int(4 * embed_dim)),
            "linear2": Variable(embed_dim, target_len),
            "add2": Variable(embed_dim, target_len),
            "layernorm2": Variable(embed_dim, target_len),
        }

        self.tunable = tunable
        self.mm = {
            "Q": ("w", "x"),
            "K": ("w", "x"),
            "V": ("w", "x"),
            "A": ("K^T", "Q"),
            "B": ("V", "A^bar"),
            "B_multi": ("W_0", "B_concat"),
            "linear1": ("W_1", "layernorm1"),
            "linear2": ("W_2", "relu"),
        }

    def get_resource(self, rsc_predictors, mixed_precision, breakdown):
        if breakdown:
            rsc = {}
            rsc["softmax"] = (
                rsc_predictors["softmax"].predict([[self.variables["A^bar"].row]])[0]
                * self.num_head
            )
            rsc["layernorm1"] = rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm1"].row]]
            )[0]
            rsc["layernorm2"] = rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm2"].row]]
            )[0]
            for k, (n1, n2) in self.mm.items():
                if mixed_precision:
                    pass
                else:
                    mm_rsc = rsc_predictors["matmul"].predict(
                        [
                            [
                                self.variables[n2].row,
                                self.variables[n1].row,
                                int(self.variables[n2].row * self.variables[n1].row),
                            ]
                        ]
                    )[0]
                if k in ["Q", "K", "V", "A", "B"]:
                    mm_rsc *= self.num_head
                rsc[k] = mm_rsc
        else:
            rsc = 0
            rsc += (
                rsc_predictors["softmax"].predict([[self.variables["A^bar"].row]])[0]
                * self.num_head
            )
            rsc += rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm1"].row]]
            )[0]
            rsc += rsc_predictors["layernorm"].predict(
                [[self.variables["layernorm2"].row]]
            )[0]
            for k, (n1, n2) in self.mm.items():
                if mixed_precision:
                    pass
                else:
                    mm_rsc = rsc_predictors["matmul"].predict(
                        [
                            [
                                self.variables[n2].row,
                                self.variables[n1].row,
                                int(self.variables[n2].row * self.variables[n1].row),
                            ]
                        ]
                    )[0]
                if k in ["Q", "K", "V", "A", "B"]:
                    mm_rsc *= self.num_head
                rsc += mm_rsc
        return rsc


class Network:
    def __init__(self):
        self.blocks = []

    def update(self, config, input_II):
        config = copy.deepcopy(config)
        assert len(config) == len(self.blocks)
        for i, block in enumerate(self.blocks):
            if i > 0:
                config[i]["x"] = list(self.blocks[i - 1].variables.values())[-1].row
                input_II = list(self.blocks[i - 1].variables.values())[-1].output_II
            block.update(config[i], input_II)

    def get_throughput(self, freq=3.5e8):
        return freq / self.blocks[-1].get_latency()

    def get_resource(self, rsc_predictors, mixed_precision=False, breakdown=False):
        if breakdown:
            rsc = []
            for block in self.blocks:
                rsc.append(
                    block.get_resource(rsc_predictors, mixed_precision, breakdown)
                )
        else:
            rsc = 0
            for block in self.blocks:
                rsc += block.get_resource(rsc_predictors, mixed_precision, breakdown)
        return rsc


class OPT125m(Network):
    def __init__(self):
        super().__init__()
        self.blocks = [
            AttentionBlockI(
                ["x", "w", "W_0", "W_1", "W_2"],
                target_len=128,
                num_head=12,
                embed_dim=768,
                head_dim=64,
            )
        ]
        self.blocks += [
            AttentionBlockI(
                ["w", "W_0", "W_1", "W_2"],
                target_len=128,
                num_head=12,
                embed_dim=768,
                head_dim=64,
            )
        ] * 11


class OPT350m(Network):
    def __init__(self):
        super().__init__()
        self.blocks = [
            AttentionBlockII(
                ["x", "w", "W_0", "W_1", "W_2"],
                target_len=128,
                num_head=16,
                embed_dim=1024,
                head_dim=64,
            )
        ]
        self.blocks += [
            AttentionBlockII(
                ["w", "W_0", "W_1", "W_2"],
                target_len=128,
                num_head=16,
                embed_dim=1024,
                head_dim=64,
            )
        ] * 23


class OPT1_3b(Network):
    def __init__(self):
        super().__init__()
        self.blocks = [
            AttentionBlockI(
                ["x", "w", "W_0", "W_1", "W_2"],
                target_len=128,
                num_head=32,
                embed_dim=2048,
                head_dim=64,
            )
        ]
        self.blocks += [
            AttentionBlockI(
                ["w", "W_0", "W_1", "W_2"],
                target_len=128,
                num_head=32,
                embed_dim=2048,
                head_dim=64,
            )
        ] * 23


class BertSmall(OPT125m):
    def load_mixed_precisoin(self, config_path):
        with open(config_path, "r") as f:
            config = toml.load(f)
            """
            :       ("w", "layernorm0"),
            "K":       ("w", "layernorm0"),
            "V":       ("w", "layernorm0"),
            "A":       ("K^T", "Q"),
            "B":       ("V", "A^bar"),
            "B_multi": ("W_0", "B_concat"),
            "linear1": ("W_1", "layernorm1"),
            "linear2": ("W_2", "relu")
            """

        def _update_quantization(variable, config):
            variable.data_in_width = config["data_in_width"]
            variable.data_in_frac_width = config["data_in_frac_width"]
            variable.weight_width = config["weight_width"]
            variable.weight_frac_width = config["weight_frac_width"]

        for i, block in enumerate(self.blocks):
            _update_quantization(
                block.variables["Q"],
                config["module_nodes_to_modify"][
                    "model.decoder.layers." + str(i) + ".self_attn.q_proj"
                ],
            )
            _update_quantization(
                block.variables["K"],
                config["module_nodes_to_modify"][
                    "model.decoder.layers." + str(i) + ".self_attn.k_proj"
                ],
            )
            _update_quantization(
                block.variables["V"],
                config["module_nodes_to_modify"][
                    "model.decoder.layers." + str(i) + ".self_attn.v_proj"
                ],
            )
            if i == 0:
                _update_quantization(
                    block.variables["A"], config["function_nodes_to_modify"]["bmm"]
                )
            else:
                _update_quantization(
                    block.variables["A"],
                    config["function_nodes_to_modify"]["bmm" + "_" + str(i * 2)],
                )
            _update_quantization(
                block.variables["B"],
                config["function_nodes_to_modify"]["bmm" + "_" + str(i * 2 + 1)],
            )
            _update_quantization(
                block.variables["B_multi"],
                config["module_nodes_to_modify"][
                    "model.decoder.layers." + str(i) + ".self_attn.out_proj"
                ],
            )
            _update_quantization(
                block.variables["linear1"],
                config["module_nodes_to_modify"][
                    "model.decoder.layers." + str(i) + ".fc1"
                ],
            )
            _update_quantization(
                block.variables["linear2"],
                config["module_nodes_to_modify"][
                    "model.decoder.layers." + str(i) + ".fc2"
                ],
            )


def pareto_frontier(Xs, Ys, Zs, maxX=False, maxY=True):
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i], Zs[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] > p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_frontZ = [pair[2] for pair in p_front]
    return p_frontX, p_frontY, p_frontZ


def main(test_name):
    def build_softmax_lut_regressor():
        features = [[1], [2], [4], [8], [16], [32]]
        lut = [624, 1213, 2383, 4715, 9323, 18531]
        reg = LinearRegression().fit(features, lut)
        return reg

    def build_softmax_ff_regressor():
        features = [[1], [2], [4], [8], [16], [32]]
        ff = [842, 1639, 3241, 6430, 12716, 25220]
        reg = LinearRegression().fit(features, ff)
        return reg

    def build_layernorm_lut_regressor():
        features = [[1], [2], [4], [8], [12], [16], [24], [32]]
        lut = [1526, 2091, 3215, 5451, 7687, 9804, 13782, 18056]
        reg = LinearRegression().fit(features, lut)
        return reg

    def build_layernorm_ff_regressor():
        features = [[1], [2], [4], [8], [12], [16], [24], [32]]
        ff = [1051, 1634, 2826, 5147, 7472, 9615, 13998, 18268]
        reg = LinearRegression().fit(features, ff)
        return reg

    def build_matmul_lut_regressor():
        features = [
            [1, 1],
            [1, 2],
            [1, 4],
            [1, 8],
            [1, 12],
            [1, 16],
            [1, 24],
            [2, 1],
            [2, 2],
            [2, 4],
            [2, 8],
            [2, 12],
            [2, 16],
            [2, 24],
            [4, 1],
            [4, 2],
            [4, 4],
            [4, 8],
            [4, 12],
            [4, 16],
            [4, 24],
            [8, 1],
            [8, 2],
            [8, 4],
            [8, 8],
            [8, 12],
            [8, 16],
            [8, 24],
            [12, 1],
            [12, 2],
            [12, 4],
            [12, 8],
            [12, 12],
            [12, 16],
            [12, 24],
            [16, 1],
            [16, 2],
            [16, 4],
            [16, 8],
            [16, 12],
            [16, 16],
            [16, 24],
            [24, 1],
            [24, 2],
            [24, 4],
            [24, 8],
            [24, 12],
            [24, 16],
            [24, 24],
        ]
        features = [[i, j, int(i * j)] for i, j in features]
        lut = [
            709,
            1043,
            1711,
            3047,
            4383,
            5749,
            8771,
            792,
            1325,
            2294,
            4214,
            6377,
            8581,
            13031,
            1099,
            1834,
            3330,
            6896,
            10569,
            14057,
            21039,
            2035,
            3655,
            6895,
            13312,
            19684,
            26001,
            38728,
            2729,
            4955,
            9427,
            18618,
            27801,
            36972,
            55137,
            3509,
            6499,
            12479,
            24186,
            35940,
            47774,
            71388,
            4959,
            9309,
            17972,
            35169,
            52386,
            69612,
            104091,
        ]
        reg = LinearRegression().fit(features, lut)
        return reg

    def build_matmul_ff_regressor():
        features = [
            [1, 1],
            [1, 2],
            [1, 4],
            [1, 8],
            [1, 12],
            [1, 16],
            [1, 24],
            [2, 1],
            [2, 2],
            [2, 4],
            [2, 8],
            [2, 12],
            [2, 16],
            [2, 24],
            [4, 1],
            [4, 2],
            [4, 4],
            [4, 8],
            [4, 12],
            [4, 16],
            [4, 24],
            [8, 1],
            [8, 2],
            [8, 4],
            [8, 8],
            [8, 12],
            [8, 16],
            [8, 24],
            [12, 1],
            [12, 2],
            [12, 4],
            [12, 8],
            [12, 12],
            [12, 16],
            [12, 24],
            [16, 1],
            [16, 2],
            [16, 4],
            [16, 8],
            [16, 12],
            [16, 16],
            [16, 24],
            [24, 1],
            [24, 2],
            [24, 4],
            [24, 8],
            [24, 12],
            [24, 16],
            [24, 24],
        ]
        features = [[i, j, int(i * j)] for i, j in features]
        ff = [
            222,
            278,
            390,
            614,
            838,
            1062,
            1574,
            195,
            336,
            496,
            816,
            1200,
            1584,
            2336,
            327,
            463,
            735,
            1343,
            1967,
            2607,
            3911,
            459,
            691,
            1155,
            2139,
            3163,
            4219,
            6299,
            715,
            1073,
            1793,
            3409,
            5121,
            6793,
            10089,
            938,
            1458,
            2482,
            4770,
            7066,
            9322,
            13818,
            1204,
            1850,
            3178,
            6098,
            8938,
            11770,
            17410,
        ]
        reg = LinearRegression().fit(features, ff)
        return reg

    def build_ms_matmul_lut_regressor():
        # microsoft floating point
        features = [
            [1, 1],
            [2, 16],
            [4, 2],
            [4, 12],
            [4, 24],
            [8, 4],
            [8, 16],
            [12, 2],
            [12, 8],
            [12, 24],
            [16, 4],
            [1, 8],
            [24, 1],
            [24, 8],
            [24, 24],
            [1, 24],
            [2, 4],
            [2, 8],
            [2, 24],
            [4, 4],
            [4, 16],
            [8, 1],
            [8, 8],
            [8, 24],
            [1, 4],
            [12, 12],
            [16, 1],
            [16, 8],
            [16, 16],
            [24, 2],
            [24, 12],
            [1, 12],
            [2, 1],
            [2, 12],
            [4, 1],
            [4, 8],
            [1, 2],
            [8, 2],
            [8, 12],
            [12, 1],
            [12, 4],
            [12, 16],
            [16, 2],
            [16, 12],
            [16, 24],
            [24, 4],
            [24, 16],
            [1, 16],
            [2, 2],
        ]
        features = [[i, j, int(i * j)] for i, j in features]
        lut = [
            550,
            5607,
            1429,
            6559,
            12554,
            3887,
            14160,
            2949,
            10443,
            30385,
            6891,
            2223,
            2860,
            18979,
            55744,
            6149,
            1647,
            2951,
            8275,
            2483,
            8522,
            1262,
            7281,
            20998,
            1267,
            15439,
            2088,
            13295,
            26106,
            5161,
            28192,
            3179,
            669,
            4261,
            902,
            4499,
            789,
            2159,
            10778,
            1700,
            5447,
            20426,
            3689,
            19735,
            38884,
            9763,
            37372,
            4171,
            995,
        ]
        reg = LinearRegression().fit(features, lut)
        return reg

    def build_ms_matmul_ff_regressor():
        # microsoft floating point
        features = [
            [1, 1],
            [2, 16],
            [4, 2],
            [4, 12],
            [4, 24],
            [8, 4],
            [8, 16],
            [12, 2],
            [12, 8],
            [12, 24],
            [16, 4],
            [1, 8],
            [24, 1],
            [24, 8],
            [24, 24],
            [1, 24],
            [2, 4],
            [2, 8],
            [2, 24],
            [4, 4],
            [4, 16],
            [8, 1],
            [8, 8],
            [8, 24],
            [1, 4],
            [12, 12],
            [16, 1],
            [16, 8],
            [16, 16],
            [24, 2],
            [24, 12],
            [1, 12],
            [2, 1],
            [2, 12],
            [4, 1],
            [4, 8],
            [1, 2],
            [8, 2],
            [8, 12],
            [12, 1],
            [12, 4],
            [12, 16],
            [16, 2],
            [16, 12],
            [16, 24],
            [24, 4],
            [24, 16],
            [1, 16],
            [2, 2],
        ]
        features = [[i, j, int(i * j)] for i, j in features]
        ff = [
            71,
            923,
            362,
            1372,
            2671,
            855,
            3078,
            623,
            1829,
            5125,
            1213,
            302,
            678,
            3037,
            8524,
            830,
            311,
            515,
            1331,
            546,
            1795,
            358,
            1596,
            4570,
            170,
            2633,
            499,
            2165,
            4104,
            1015,
            4400,
            434,
            158,
            719,
            270,
            954,
            104,
            517,
            2317,
            424,
            1025,
            3462,
            737,
            3117,
            6058,
            1689,
            5778,
            566,
            209,
        ]

        reg = LinearRegression().fit(features, ff)
        return reg

    def build_mp_matmul_lut_regressor():
        # mixed_precision
        features = [
            [1, 1, 2, 1],
            [1, 1, 3, 1],
            [1, 1, 4, 1],
            [1, 1, 5, 1],
            [1, 1, 6, 1],
            [1, 1, 7, 1],
            [1, 1, 8, 1],
            [1, 2, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 3, 1],
            [1, 2, 4, 1],
            [1, 2, 5, 1],
            [1, 2, 6, 1],
            [1, 2, 7, 1],
            [1, 2, 8, 1],
            [1, 4, 1, 1],
            [1, 4, 2, 1],
            [1, 4, 3, 1],
            [1, 4, 4, 1],
            [1, 4, 5, 1],
            [1, 4, 6, 1],
            [1, 4, 7, 1],
            [1, 4, 8, 1],
            [1, 8, 1, 1],
            [1, 8, 2, 1],
            [1, 8, 3, 1],
            [1, 8, 4, 1],
            [1, 8, 5, 1],
            [1, 8, 6, 1],
            [1, 8, 7, 1],
            [1, 8, 8, 1],
            [2, 1, 1, 1],
            [2, 1, 2, 1],
            [2, 1, 3, 1],
            [2, 1, 4, 1],
            [2, 1, 5, 1],
            [2, 1, 6, 1],
            [2, 1, 7, 1],
            [2, 1, 8, 1],
            [2, 2, 1, 1],
            [2, 2, 2, 1],
            [2, 2, 3, 1],
            [2, 2, 4, 1],
            [2, 2, 5, 1],
            [2, 2, 6, 1],
            [2, 2, 7, 1],
            [2, 2, 8, 1],
            [2, 4, 1, 1],
            [2, 4, 2, 1],
            [2, 4, 3, 1],
            [2, 4, 4, 1],
            [2, 4, 5, 1],
            [2, 4, 6, 1],
            [2, 4, 7, 1],
            [2, 4, 8, 1],
            [2, 8, 1, 1],
            [2, 8, 2, 1],
            [2, 8, 3, 1],
            [2, 8, 4, 1],
            [2, 8, 5, 1],
            [2, 8, 6, 1],
            [2, 8, 7, 1],
            [2, 8, 8, 1],
            [4, 1, 1, 1],
            [4, 1, 2, 1],
            [4, 1, 3, 1],
            [4, 1, 4, 1],
            [4, 1, 5, 1],
            [4, 1, 6, 1],
            [4, 1, 7, 1],
            [4, 1, 8, 1],
            [4, 2, 1, 1],
            [4, 2, 2, 1],
            [4, 2, 3, 1],
            [4, 2, 4, 1],
            [4, 2, 5, 1],
            [4, 2, 6, 1],
            [4, 2, 7, 1],
            [4, 2, 8, 1],
            [4, 4, 1, 1],
            [4, 4, 2, 1],
            [4, 4, 3, 1],
            [4, 4, 4, 1],
            [4, 4, 5, 1],
            [4, 4, 6, 1],
            [4, 4, 7, 1],
            [4, 4, 8, 1],
            [4, 8, 1, 1],
            [4, 8, 2, 1],
            [4, 8, 3, 1],
            [4, 8, 4, 1],
            [4, 8, 5, 1],
            [4, 8, 6, 1],
            [4, 8, 7, 1],
            [4, 8, 8, 1],
            [8, 1, 1, 1],
            [8, 1, 2, 1],
            [8, 1, 3, 1],
            [8, 1, 4, 1],
            [8, 1, 5, 1],
            [8, 1, 6, 1],
            [8, 1, 7, 1],
            [8, 1, 8, 1],
            [8, 2, 1, 1],
            [8, 2, 2, 1],
            [8, 2, 3, 1],
            [8, 2, 4, 1],
            [8, 2, 5, 1],
            [8, 2, 6, 1],
            [8, 2, 7, 1],
            [8, 2, 8, 1],
            [8, 4, 1, 1],
            [8, 4, 2, 1],
            [8, 4, 3, 1],
            [8, 4, 4, 1],
            [8, 4, 5, 1],
            [8, 4, 6, 1],
            [8, 4, 7, 1],
            [8, 4, 8, 1],
            [8, 8, 1, 1],
            [8, 8, 2, 1],
            [8, 8, 3, 1],
            [8, 8, 4, 1],
            [8, 8, 5, 1],
            [8, 8, 6, 1],
            [8, 8, 7, 1],
            [8, 8, 8, 1],
            [1, 1, 1, 2],
            [1, 1, 2, 2],
            [1, 1, 3, 2],
            [1, 1, 4, 2],
            [1, 1, 5, 2],
            [1, 1, 6, 2],
            [1, 1, 7, 2],
            [1, 1, 8, 2],
            [1, 2, 1, 2],
            [1, 2, 2, 2],
            [1, 2, 3, 2],
            [1, 2, 4, 2],
            [1, 2, 5, 2],
            [1, 2, 6, 2],
            [1, 2, 7, 2],
            [1, 2, 8, 2],
            [1, 4, 1, 2],
            [1, 4, 2, 2],
            [1, 4, 3, 2],
            [1, 4, 4, 2],
            [1, 4, 5, 2],
            [1, 4, 6, 2],
            [1, 4, 7, 2],
            [1, 4, 8, 2],
            [1, 8, 1, 2],
            [1, 8, 2, 2],
            [1, 8, 3, 2],
            [1, 8, 4, 2],
            [1, 8, 5, 2],
            [1, 8, 6, 2],
            [1, 8, 7, 2],
            [1, 8, 8, 2],
            [2, 1, 1, 2],
            [2, 1, 2, 2],
            [2, 1, 3, 2],
            [2, 1, 4, 2],
            [2, 1, 5, 2],
            [2, 1, 6, 2],
            [2, 1, 7, 2],
            [2, 1, 8, 2],
            [2, 2, 1, 2],
            [2, 2, 2, 2],
            [2, 2, 3, 2],
            [2, 2, 4, 2],
            [2, 2, 5, 2],
            [2, 2, 6, 2],
            [2, 2, 7, 2],
            [2, 2, 8, 2],
            [2, 4, 1, 2],
            [2, 4, 2, 2],
            [2, 4, 3, 2],
            [2, 4, 4, 2],
            [2, 4, 5, 2],
            [2, 4, 6, 2],
            [2, 4, 7, 2],
            [2, 4, 8, 2],
            [2, 8, 1, 2],
            [2, 8, 2, 2],
            [2, 8, 3, 2],
            [2, 8, 4, 2],
            [2, 8, 5, 2],
            [2, 8, 6, 2],
            [2, 8, 7, 2],
            [2, 8, 8, 2],
            [4, 1, 1, 2],
            [4, 1, 2, 2],
            [4, 1, 3, 2],
            [4, 1, 4, 2],
            [4, 1, 5, 2],
            [4, 1, 6, 2],
            [4, 1, 7, 2],
            [4, 1, 8, 2],
            [4, 2, 1, 2],
            [4, 2, 2, 2],
            [4, 2, 3, 2],
            [4, 2, 4, 2],
            [4, 2, 5, 2],
            [4, 2, 6, 2],
            [4, 2, 7, 2],
            [4, 2, 8, 2],
            [4, 4, 1, 2],
            [4, 4, 2, 2],
            [4, 4, 3, 2],
            [4, 4, 4, 2],
            [4, 4, 5, 2],
            [4, 4, 6, 2],
            [4, 4, 7, 2],
            [4, 4, 8, 2],
            [4, 8, 1, 2],
            [4, 8, 2, 2],
            [4, 8, 3, 2],
            [4, 8, 4, 2],
            [4, 8, 5, 2],
            [4, 8, 6, 2],
            [4, 8, 7, 2],
            [4, 8, 8, 2],
            [8, 1, 1, 2],
            [8, 1, 2, 2],
            [8, 1, 3, 2],
            [8, 1, 4, 2],
            [8, 1, 5, 2],
            [8, 1, 6, 2],
            [8, 1, 7, 2],
            [8, 1, 8, 2],
            [8, 2, 1, 2],
            [8, 2, 2, 2],
            [8, 2, 3, 2],
            [8, 2, 4, 2],
            [8, 2, 5, 2],
            [8, 2, 6, 2],
            [8, 2, 7, 2],
            [8, 2, 8, 2],
            [8, 4, 1, 2],
            [8, 4, 2, 2],
            [8, 4, 3, 2],
            [8, 4, 4, 2],
            [8, 4, 5, 2],
            [8, 4, 6, 2],
            [8, 4, 7, 2],
            [8, 4, 8, 2],
            [8, 8, 1, 2],
            [8, 8, 2, 2],
            [8, 8, 3, 2],
            [8, 8, 4, 2],
            [8, 8, 5, 2],
            [8, 8, 6, 2],
            [1, 1, 1, 3],
            [1, 1, 2, 3],
            [1, 1, 3, 3],
            [1, 1, 4, 3],
            [1, 1, 5, 3],
            [1, 1, 6, 3],
            [1, 1, 7, 3],
            [1, 1, 8, 3],
            [1, 2, 1, 3],
            [1, 2, 2, 3],
            [1, 2, 3, 3],
            [1, 2, 4, 3],
            [1, 2, 5, 3],
            [1, 2, 6, 3],
            [1, 2, 7, 3],
            [1, 2, 8, 3],
            [1, 4, 1, 3],
            [1, 4, 2, 3],
            [1, 4, 3, 3],
            [1, 4, 4, 3],
            [1, 4, 5, 3],
            [1, 4, 6, 3],
            [1, 4, 7, 3],
            [1, 4, 8, 3],
            [1, 8, 1, 3],
            [1, 8, 2, 3],
            [1, 8, 3, 3],
            [1, 8, 4, 3],
            [1, 8, 5, 3],
            [1, 8, 6, 3],
            [1, 8, 7, 3],
            [1, 8, 8, 3],
            [2, 1, 1, 3],
            [2, 1, 2, 3],
            [2, 1, 3, 3],
            [2, 1, 4, 3],
            [2, 1, 5, 3],
            [2, 1, 6, 3],
            [2, 1, 7, 3],
            [2, 1, 8, 3],
            [2, 2, 1, 3],
            [2, 2, 2, 3],
            [2, 2, 3, 3],
            [2, 2, 4, 3],
            [2, 2, 5, 3],
            [2, 2, 6, 3],
            [2, 2, 7, 3],
            [2, 2, 8, 3],
            [2, 4, 1, 3],
            [2, 4, 2, 3],
            [2, 4, 3, 3],
            [2, 4, 4, 3],
            [2, 4, 5, 3],
            [2, 4, 6, 3],
            [2, 4, 7, 3],
            [2, 4, 8, 3],
            [2, 8, 1, 3],
            [2, 8, 2, 3],
            [2, 8, 3, 3],
            [2, 8, 4, 3],
            [2, 8, 5, 3],
            [2, 8, 6, 3],
            [2, 8, 7, 3],
            [2, 8, 8, 3],
            [4, 1, 1, 3],
            [4, 1, 2, 3],
            [4, 1, 3, 3],
            [4, 1, 4, 3],
            [4, 1, 5, 3],
            [4, 1, 6, 3],
            [4, 1, 7, 3],
            [4, 1, 8, 3],
            [4, 2, 1, 3],
            [4, 2, 2, 3],
            [4, 2, 3, 3],
            [4, 2, 4, 3],
            [4, 2, 5, 3],
            [4, 2, 6, 3],
            [4, 2, 7, 3],
            [4, 2, 8, 3],
            [4, 4, 1, 3],
            [4, 4, 2, 3],
            [4, 4, 3, 3],
            [4, 4, 4, 3],
            [4, 4, 5, 3],
            [4, 4, 6, 3],
            [4, 4, 7, 3],
            [4, 4, 8, 3],
            [4, 8, 1, 3],
            [4, 8, 2, 3],
            [4, 8, 3, 3],
            [4, 8, 4, 3],
            [4, 8, 5, 3],
            [4, 8, 6, 3],
            [4, 8, 7, 3],
            [4, 8, 8, 3],
            [8, 1, 1, 3],
            [8, 1, 2, 3],
            [8, 1, 3, 3],
            [8, 1, 4, 3],
            [8, 1, 5, 3],
            [8, 1, 6, 3],
            [8, 1, 7, 3],
            [8, 1, 8, 3],
            [8, 2, 1, 3],
            [8, 2, 2, 3],
            [8, 2, 3, 3],
            [8, 2, 4, 3],
            [8, 2, 5, 3],
            [8, 2, 6, 3],
            [8, 2, 7, 3],
            [8, 2, 8, 3],
            [8, 4, 1, 3],
            [8, 4, 2, 3],
            [8, 4, 3, 3],
            [8, 4, 4, 3],
            [8, 4, 5, 3],
            [8, 4, 6, 3],
            [8, 4, 7, 3],
            [8, 4, 8, 3],
            [8, 8, 1, 3],
            [8, 8, 2, 3],
            [8, 8, 3, 3],
            [8, 8, 4, 3],
            [8, 8, 5, 3],
            [8, 8, 6, 3],
            [8, 8, 7, 3],
            [8, 8, 8, 3],
            [1, 1, 1, 4],
            [1, 1, 2, 4],
            [1, 1, 3, 4],
            [1, 1, 4, 4],
            [1, 1, 5, 4],
            [1, 1, 6, 4],
            [1, 1, 7, 4],
            [1, 1, 8, 4],
            [1, 2, 1, 4],
            [1, 2, 2, 4],
            [1, 2, 3, 4],
            [1, 2, 4, 4],
            [1, 2, 5, 4],
            [1, 2, 6, 4],
            [1, 2, 7, 4],
            [1, 2, 8, 4],
            [1, 4, 1, 4],
            [1, 4, 2, 4],
            [1, 4, 3, 4],
            [1, 4, 4, 4],
            [1, 4, 5, 4],
            [1, 4, 6, 4],
            [1, 4, 7, 4],
            [1, 4, 8, 4],
            [1, 8, 1, 4],
            [1, 8, 2, 4],
            [1, 8, 3, 4],
            [1, 8, 4, 4],
            [1, 8, 5, 4],
            [1, 8, 6, 4],
            [1, 8, 7, 4],
            [1, 8, 8, 4],
            [2, 1, 1, 4],
            [2, 1, 2, 4],
            [2, 1, 3, 4],
            [2, 1, 4, 4],
            [2, 1, 5, 4],
            [2, 1, 6, 4],
            [2, 1, 7, 4],
            [2, 1, 8, 4],
            [2, 2, 1, 4],
            [2, 2, 2, 4],
            [2, 2, 3, 4],
            [2, 2, 4, 4],
            [2, 2, 5, 4],
            [2, 2, 6, 4],
            [2, 2, 7, 4],
            [2, 2, 8, 4],
            [2, 4, 1, 4],
            [2, 4, 2, 4],
            [2, 4, 3, 4],
            [2, 4, 4, 4],
            [2, 4, 5, 4],
            [2, 4, 6, 4],
            [2, 4, 7, 4],
            [2, 4, 8, 4],
            [2, 8, 1, 4],
            [2, 8, 2, 4],
            [2, 8, 3, 4],
            [2, 8, 4, 4],
            [2, 8, 5, 4],
            [2, 8, 6, 4],
            [2, 8, 7, 4],
            [2, 8, 8, 4],
            [4, 1, 1, 4],
            [4, 1, 2, 4],
            [4, 1, 3, 4],
            [4, 1, 4, 4],
            [4, 1, 5, 4],
            [4, 1, 6, 4],
            [4, 1, 7, 4],
            [4, 1, 8, 4],
            [4, 2, 1, 4],
            [4, 2, 2, 4],
            [4, 2, 3, 4],
            [4, 2, 4, 4],
            [4, 2, 5, 4],
            [4, 2, 6, 4],
            [4, 2, 7, 4],
            [4, 2, 8, 4],
            [4, 4, 1, 4],
            [4, 4, 2, 4],
            [4, 4, 3, 4],
            [4, 4, 4, 4],
            [4, 4, 5, 4],
            [4, 4, 6, 4],
            [4, 4, 7, 4],
            [4, 4, 8, 4],
            [4, 8, 1, 4],
            [4, 8, 2, 4],
            [4, 8, 3, 4],
            [4, 8, 4, 4],
            [4, 8, 5, 4],
            [4, 8, 6, 4],
            [4, 8, 7, 4],
            [4, 8, 8, 4],
            [8, 1, 1, 4],
            [8, 1, 2, 4],
            [8, 1, 3, 4],
            [8, 1, 4, 4],
            [8, 1, 5, 4],
            [8, 1, 6, 4],
            [8, 1, 7, 4],
            [8, 1, 8, 4],
            [8, 2, 1, 4],
            [8, 2, 2, 4],
            [8, 2, 3, 4],
            [8, 2, 4, 4],
            [8, 2, 5, 4],
            [8, 2, 6, 4],
            [8, 2, 7, 4],
            [8, 2, 8, 4],
            [8, 4, 1, 4],
            [8, 4, 2, 4],
            [8, 4, 3, 4],
            [8, 4, 4, 4],
            [8, 4, 5, 4],
            [8, 4, 6, 4],
            [8, 4, 7, 4],
            [8, 4, 8, 4],
            [8, 8, 1, 4],
            [8, 8, 2, 4],
            [8, 8, 3, 4],
            [8, 8, 4, 4],
            [8, 8, 5, 4],
            [8, 8, 6, 4],
            [8, 8, 7, 4],
            [8, 8, 8, 4],
            [1, 1, 1, 5],
            [1, 1, 2, 5],
            [1, 1, 3, 5],
            [1, 1, 4, 5],
            [1, 1, 5, 5],
            [1, 1, 6, 5],
            [1, 1, 7, 5],
            [1, 1, 8, 5],
            [1, 2, 1, 5],
            [1, 2, 2, 5],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 2, 5, 5],
            [1, 2, 6, 5],
            [1, 2, 7, 5],
            [1, 2, 8, 5],
            [1, 4, 1, 5],
            [1, 4, 2, 5],
            [1, 4, 3, 5],
            [1, 4, 4, 5],
            [1, 4, 5, 5],
            [1, 4, 6, 5],
            [1, 4, 7, 5],
            [1, 4, 8, 5],
            [1, 8, 1, 5],
            [1, 8, 2, 5],
            [1, 8, 3, 5],
            [1, 8, 4, 5],
            [1, 8, 5, 5],
            [1, 8, 6, 5],
            [1, 8, 7, 5],
            [1, 8, 8, 5],
            [2, 1, 1, 5],
            [2, 1, 2, 5],
            [2, 1, 3, 5],
            [2, 1, 4, 5],
            [2, 1, 5, 5],
            [2, 1, 6, 5],
            [2, 1, 7, 5],
            [2, 1, 8, 5],
            [2, 2, 1, 5],
            [2, 2, 2, 5],
            [2, 2, 3, 5],
            [2, 2, 4, 5],
            [2, 2, 5, 5],
            [2, 2, 6, 5],
            [2, 2, 7, 5],
            [2, 2, 8, 5],
            [2, 4, 1, 5],
            [2, 4, 2, 5],
            [2, 4, 3, 5],
            [2, 4, 4, 5],
            [2, 4, 5, 5],
            [2, 4, 6, 5],
            [2, 4, 7, 5],
            [2, 4, 8, 5],
            [2, 8, 1, 5],
            [2, 8, 2, 5],
            [2, 8, 3, 5],
            [2, 8, 4, 5],
            [2, 8, 5, 5],
            [2, 8, 6, 5],
            [2, 8, 7, 5],
            [2, 8, 8, 5],
            [4, 1, 1, 5],
            [4, 1, 2, 5],
            [4, 1, 3, 5],
            [4, 1, 4, 5],
            [4, 1, 5, 5],
            [4, 1, 6, 5],
            [4, 1, 7, 5],
            [4, 1, 8, 5],
            [4, 2, 1, 5],
            [4, 2, 2, 5],
            [4, 2, 3, 5],
            [4, 2, 4, 5],
            [4, 2, 5, 5],
            [4, 2, 6, 5],
            [4, 2, 7, 5],
            [4, 2, 8, 5],
            [4, 4, 1, 5],
            [4, 4, 2, 5],
            [4, 4, 3, 5],
            [4, 4, 4, 5],
            [4, 4, 5, 5],
            [4, 4, 6, 5],
            [4, 4, 7, 5],
            [4, 4, 8, 5],
            [4, 8, 1, 5],
            [4, 8, 2, 5],
            [4, 8, 3, 5],
            [4, 8, 4, 5],
            [4, 8, 5, 5],
            [4, 8, 6, 5],
            [4, 8, 7, 5],
            [4, 8, 8, 5],
            [8, 1, 1, 5],
            [8, 1, 2, 5],
            [8, 1, 3, 5],
            [8, 1, 4, 5],
            [8, 1, 5, 5],
            [8, 1, 6, 5],
            [8, 1, 7, 5],
            [8, 1, 8, 5],
            [8, 2, 1, 5],
            [8, 2, 2, 5],
            [8, 2, 3, 5],
            [8, 2, 4, 5],
            [8, 2, 5, 5],
            [8, 2, 6, 5],
            [8, 2, 7, 5],
            [8, 2, 8, 5],
            [8, 4, 1, 5],
            [8, 4, 2, 5],
            [8, 4, 3, 5],
            [8, 4, 4, 5],
            [8, 4, 5, 5],
            [8, 4, 6, 5],
            [8, 4, 7, 5],
            [8, 4, 8, 5],
            [8, 8, 1, 5],
            [8, 8, 2, 5],
            [8, 8, 3, 5],
            [8, 8, 4, 5],
            [8, 8, 5, 5],
            [8, 8, 6, 5],
            [8, 8, 7, 5],
            [8, 8, 8, 5],
            [1, 1, 1, 6],
            [1, 1, 2, 6],
            [1, 1, 3, 6],
            [1, 1, 4, 6],
            [1, 1, 5, 6],
            [1, 1, 6, 6],
            [1, 1, 7, 6],
            [1, 1, 8, 6],
            [1, 2, 1, 6],
            [1, 2, 2, 6],
            [1, 2, 3, 6],
            [1, 2, 4, 6],
            [1, 2, 5, 6],
            [1, 2, 6, 6],
            [1, 2, 7, 6],
            [1, 2, 8, 6],
            [1, 4, 1, 6],
            [1, 4, 2, 6],
            [1, 4, 3, 6],
            [1, 4, 4, 6],
            [1, 4, 5, 6],
            [1, 4, 6, 6],
            [1, 4, 7, 6],
            [1, 4, 8, 6],
            [1, 8, 1, 6],
            [1, 8, 2, 6],
            [1, 8, 3, 6],
            [1, 8, 4, 6],
            [1, 8, 5, 6],
            [1, 8, 6, 6],
            [1, 8, 7, 6],
            [1, 8, 8, 6],
            [2, 1, 1, 6],
            [2, 1, 2, 6],
            [2, 1, 3, 6],
            [2, 1, 4, 6],
            [2, 1, 5, 6],
            [2, 1, 6, 6],
            [2, 1, 7, 6],
            [2, 1, 8, 6],
            [2, 2, 1, 6],
            [2, 2, 2, 6],
            [2, 2, 3, 6],
            [2, 2, 4, 6],
            [2, 2, 5, 6],
            [2, 2, 6, 6],
            [2, 2, 7, 6],
            [2, 2, 8, 6],
            [2, 4, 1, 6],
            [2, 4, 2, 6],
            [2, 4, 3, 6],
            [2, 4, 4, 6],
            [2, 4, 5, 6],
            [2, 4, 6, 6],
            [2, 4, 7, 6],
            [2, 4, 8, 6],
            [2, 8, 1, 6],
            [2, 8, 2, 6],
            [2, 8, 3, 6],
            [2, 8, 4, 6],
            [2, 8, 5, 6],
            [2, 8, 6, 6],
            [2, 8, 7, 6],
            [2, 8, 8, 6],
            [4, 1, 1, 6],
            [4, 1, 2, 6],
            [4, 1, 3, 6],
            [4, 1, 4, 6],
            [4, 1, 5, 6],
            [4, 1, 6, 6],
            [4, 1, 7, 6],
            [4, 1, 8, 6],
            [4, 2, 1, 6],
            [4, 2, 2, 6],
            [4, 2, 3, 6],
            [4, 2, 4, 6],
            [4, 2, 5, 6],
            [4, 2, 6, 6],
            [4, 2, 7, 6],
            [4, 2, 8, 6],
        ]
        features = [
            [
                i,
                j,
                m,
                n,
                int(i * j),
                int(i * m),
                int(i * n),
                int(j * m),
                int(j * n),
                int(m * n),
                int(i * j * m),
                int(i * j * n),
                int(i * m * n),
                int(j * m * n),
                int(i * j * m * n),
            ]
            for i, j, m, n in features
        ]
        lut = [
            474,
            509,
            553,
            583,
            645,
            679,
            726,
            600,
            626,
            696,
            784,
            844,
            936,
            1004,
            1098,
            850,
            930,
            1070,
            1246,
            1366,
            1518,
            1654,
            1842,
            1350,
            1538,
            1818,
            2170,
            2410,
            2682,
            2954,
            3330,
            508,
            519,
            574,
            646,
            720,
            769,
            822,
            947,
            664,
            716,
            826,
            1002,
            1086,
            1184,
            1290,
            1508,
            976,
            1110,
            1330,
            1618,
            1818,
            2014,
            2226,
            2630,
            1600,
            1898,
            2338,
            2946,
            3282,
            3674,
            4098,
            4874,
            574,
            609,
            704,
            864,
            934,
            1076,
            1159,
            1254,
            792,
            896,
            1086,
            1374,
            1514,
            1766,
            1932,
            2160,
            1228,
            1470,
            1850,
            2394,
            2674,
            3146,
            3478,
            3972,
            2100,
            2624,
            3384,
            4440,
            5218,
            6234,
            7074,
            8589,
            706,
            803,
            976,
            1246,
            1433,
            1592,
            1738,
            1986,
            1048,
            1284,
            1630,
            2138,
            2490,
            2756,
            3046,
            3544,
            1732,
            2246,
            2938,
            3922,
            4792,
            4940,
            5518,
            6555,
            3124,
            4200,
            5584,
            7520,
            9355,
            9272,
            10427,
            12536,
            472,
            456,
            489,
            527,
            569,
            600,
            664,
            696,
            620,
            590,
            656,
            732,
            816,
            910,
            974,
            1038,
            916,
            858,
            990,
            1142,
            1310,
            1434,
            1594,
            1722,
            1508,
            1394,
            1658,
            1962,
            2298,
            2546,
            2834,
            3090,
            515,
            483,
            534,
            594,
            696,
            743,
            792,
            841,
            704,
            644,
            746,
            866,
            1038,
            1132,
            1230,
            1328,
            1082,
            966,
            1170,
            1410,
            1722,
            1910,
            2106,
            2302,
            1838,
            1610,
            2018,
            2498,
            3090,
            3466,
            3858,
            4250,
            601,
            537,
            624,
            728,
            886,
            965,
            1089,
            1142,
            872,
            752,
            926,
            1134,
            1418,
            1576,
            1792,
            1909,
            1414,
            1182,
            1530,
            1946,
            2482,
            2798,
            3207,
            3469,
            2498,
            2042,
            2738,
            3570,
            4610,
            5434,
            6428,
            7163,
            787,
            659,
            816,
            1038,
            1270,
            1507,
            1674,
            1802,
            1236,
            996,
            1310,
            1722,
            2186,
            2625,
            2927,
            3185,
            2134,
            1670,
            2298,
            3058,
            4018,
            5042,
            5848,
            6530,
            3954,
            3042,
            4298,
            5850,
            7706,
            9744,
            509,
            491,
            491,
            529,
            571,
            602,
            666,
            698,
            694,
            660,
            660,
            736,
            820,
            914,
            978,
            1042,
            1064,
            998,
            998,
            1150,
            1318,
            1442,
            1602,
            1730,
            1804,
            1674,
            1674,
            1978,
            2314,
            2562,
            2850,
            3106,
            574,
            538,
            538,
            598,
            700,
            747,
            796,
            845,
            854,
            754,
            754,
            874,
            1046,
            1140,
            1238,
            1336,
            1318,
            1186,
            1186,
            1426,
            1738,
            1926,
            2122,
            2318,
            2310,
            2050,
            2050,
            2530,
            3122,
            3498,
            3890,
            4282,
            704,
            632,
            632,
            736,
            894,
            973,
            1097,
            1150,
            1110,
            942,
            942,
            1150,
            1434,
            1592,
            1808,
            1925,
            1826,
            1562,
            1562,
            1978,
            2514,
            2830,
            3239,
            3501,
            3258,
            2802,
            2802,
            3634,
            4674,
            5498,
            6492,
            7227,
            982,
            838,
            838,
            1060,
            1292,
            1523,
            1690,
            1818,
            1658,
            1342,
            1342,
            1754,
            2218,
            2657,
            2959,
            3217,
            2890,
            2362,
            2362,
            3122,
            4082,
            5106,
            5912,
            6594,
            5380,
            4426,
            4426,
            5978,
            7834,
            9872,
            11697,
            13308,
            538,
            529,
            529,
            529,
            571,
            602,
            666,
            698,
            752,
            736,
            736,
            736,
            820,
            914,
            978,
            1042,
            1180,
            1150,
            1150,
            1150,
            1318,
            1442,
            1602,
            1730,
            2036,
            1978,
            1978,
            1978,
            2314,
            2562,
            2850,
            3106,
            648,
            598,
            598,
            598,
            700,
            747,
            796,
            845,
            938,
            874,
            874,
            874,
            1046,
            1140,
            1238,
            1336,
            1518,
            1426,
            1426,
            1426,
            1738,
            1926,
            2122,
            2318,
            2678,
            2530,
            2530,
            2530,
            3122,
            3498,
            3890,
            4282,
            804,
            736,
            736,
            736,
            894,
            973,
            1097,
            1150,
            1246,
            1150,
            1150,
            1150,
            1434,
            1592,
            1808,
            1925,
            2130,
            1978,
            1978,
            1978,
            2514,
            2830,
            3239,
            3501,
            3898,
            3634,
            3634,
            3634,
            4674,
            5498,
            6492,
            7227,
            1126,
            1054,
            1054,
            1054,
            1286,
            1523,
            1690,
            1818,
            1882,
            1754,
            1754,
            1754,
            2218,
            2657,
            2959,
            3217,
            3394,
            3122,
            3122,
            3122,
            4082,
            5106,
            5912,
            6594,
            6418,
            5978,
            5978,
            5978,
            7834,
            9872,
            11697,
            13308,
            605,
            575,
            575,
            575,
            575,
            606,
            670,
            702,
            854,
            828,
            828,
            828,
            828,
            922,
            986,
            1050,
            1352,
            1334,
            1334,
            1334,
            1334,
            1458,
            1618,
            1746,
            2348,
            2346,
            2346,
            2346,
            2346,
            2594,
            2882,
            3138,
            704,
            708,
            708,
            708,
            708,
            755,
            804,
            853,
            1050,
            1062,
            1062,
            1062,
            1062,
            1156,
            1254,
            1352,
            1742,
            1770,
            1770,
            1770,
            1770,
            1958,
            2154,
            2350,
            3126,
            3186,
            3186,
            3186,
            3186,
            3562,
            3954,
            4346,
            902,
            910,
            910,
            910,
            910,
            989,
            1113,
            1166,
            1442,
            1466,
            1466,
            1466,
            1466,
            1624,
            1840,
            1957,
            2522,
            2578,
            2578,
            2578,
            2578,
            2894,
            3303,
            3565,
            4682,
            4802,
            4802,
            4802,
            4802,
            5626,
            6620,
            7355,
            1352,
            1318,
            1318,
            1318,
            1318,
            1555,
            1722,
            1850,
            2266,
            2282,
            2282,
            2282,
            2282,
            2721,
            3023,
            3281,
            4130,
            4210,
            4210,
            4210,
            4210,
            5234,
            6040,
            6722,
            7978,
            8090,
            8090,
            8090,
            8090,
            10128,
            11953,
            13564,
            637,
            606,
            606,
            606,
            606,
            606,
            670,
            702,
            918,
            922,
            922,
            922,
            922,
            922,
            986,
            1050,
            1480,
            1458,
            1458,
            1458,
            1458,
            1458,
            1618,
            1746,
            2604,
            2594,
            2594,
            2594,
            2594,
            2594,
            2882,
            3138,
            792,
            755,
            755,
            755,
            755,
            755,
            804,
            853,
            1194,
            1156,
            1156,
            1156,
            1156,
            1156,
            1254,
            1352,
            1998,
            1958,
            1958,
            1958,
            1958,
            1958,
            2154,
            2350,
            3606,
            3562,
            3562,
            3562,
            3562,
            3562,
            3954,
            4346,
            970,
            989,
            989,
            989,
            989,
            989,
            1113,
            1166,
            1672,
            1624,
            1624,
            1624,
            1624,
            1624,
            1840,
            1957,
        ]
        reg = LinearRegression().fit(features, lut)
        return reg

    def build_mp_matmul_ff_regressor():
        pass

    regressor = build_matmul_lut_regressor()
    print(regressor.predict([[64, 2000, 2000 * 64]]))
    # print(regressor.predict([[256,256,256*256]]))
    print("")
    # print(regressor.predict([[256,96,96*256]]))
    # print(regressor.predict([[64,192,64*192]]))
    # print(regressor.predict([[192,256,192*256]]))
    # print(regressor.predict([[256,192,256*192]]))

    """
    print(regressor.predict([[64,96,64*96]]))
    print(regressor.predict([[96,192,96*192]]))
    print(regressor.predict([[192,96,96*192]]))
    print(regressor.predict([[96,64,96*64]]))

    print(regressor.predict([[64,128,64*128]]))
    print(regressor.predict([[128,256,128*256]]))
    print(regressor.predict([[256,128,256*128]]))
    print(regressor.predict([[128,64,128*64]]))  

    print(regressor.predict([[64, 128,128*64]]))    
    print(regressor.predict([[128,256,128*256]]))
    print(regressor.predict([[256,128,256*128]]))

    print(regressor.predict([[64, 512,512*64]]))  
    print(regressor.predict([[512,256,512*256]]))
    print(regressor.predict([[256,512,256*512]]))
    print(regressor.predict([[512,64,512*64]]))
    """

    rsc_predictors = {
        "softmax": build_softmax_lut_regressor(),
        "layernorm": build_layernorm_lut_regressor(),
        "matmul": build_mp_matmul_lut_regressor(),
    }

    # OPT125m
    if test_name == "OPT125m":
        net = OPT125m()
        config = [{"x": 32, "w": 64, "W_0": 96, "W_1": 192, "W_2": 96}]
        config += [{"w": 64, "W_0": 96, "W_1": 192, "W_2": 96}] * 11
    # OPT350m
    elif test_name == "OPT350m":
        net = OPT350m()
        config = [{"x": 32, "w": 64, "W_0": 128, "W_1": 256, "W_2": 128}]
        config += [{"w": 64, "W_0": 128, "W_1": 256, "W_2": 128}] * 23
    elif test_name == "OPT1_3b":
        # OPT1_3b
        net = OPT1_3b()
        config = [{"x": 32, "w": 64, "W_0": 512, "W_1": 256, "W_2": 512}]
        config += [{"w": 64, "W_0": 512, "W_1": 256, "W_2": 512}] * 23
    elif test_name == "BertSmall":
        net = BertSmall()
        config = [{"x": 32, "w": 64, "W_0": 96, "W_1": 192, "W_2": 96}]
        config += [{"w": 64, "W_0": 96, "W_1": 192, "W_2": 96}] * 11
        names = os.listdir(os.path.expanduser("~/Desktop/q_configs/"))
        names = sorted(names)
        results = []
        for name in names:
            print(name)
            net.load_mixed_precisoin(os.path.expanduser("~/Desktop/q_configs/" + name))
            net.update(config, 1)
            thr = net.get_throughput()
            rsc = net.get_resource(rsc_predictors, mixed_precision=True)
            results.append(rsc)
            print(thr)
            print(rsc)
        print(results)

    net.update(config, 1)
    thr = net.get_throughput()
    rsc = net.get_resource(rsc_predictors, mixed_precision=True)
    # l_rsc = net.get_resource(rsc_predictors, breakdown=True)
    print(thr)
    print(rsc)
    # print(l_rsc)

    throughput_results = [thr]
    resource_results = [rsc]
    config_results = [copy.deepcopy(config)]
    brakdown_results = [l_rsc]
    space_keys = []
    for i in reversed(range(len(config))):
        for j in reversed(config[i].keys()):
            space_keys.append((i, j))

    for k, (i, j) in enumerate(space_keys):
        while True:
            feasible = net.blocks[i].variables[j].get_row_feasible()
            feasible = [f for f in feasible if f < config[i][j]]
            if len(feasible) == 0:
                break
            else:
                config[i][j] = feasible[-1]
                net.update(config, 1)
                curr_thr = net.get_throughput()
                curr_rsc = net.get_resource(rsc_predictors)
                throughput_results.append(curr_thr)
                resource_results.append(curr_rsc)
                l_rsc = net.get_resource(rsc_predictors, breakdown=True)
                brakdown_results.append(l_rsc)
                config_results.append(copy.deepcopy(config))
                if curr_thr < thr:
                    for p, (m, n) in enumerate(space_keys[k + 1 :]):
                        while True:
                            feasible = net.blocks[m].variables[n].get_row_feasible()
                            feasible = [f for f in feasible if f < config[m][n]]
                            if len(feasible) == 0:
                                break
                            else:
                                config_copy = copy.deepcopy(config)
                                config_copy[m][n] = feasible[-1]
                                net.update(config_copy, 1)
                                next_thr = net.get_throughput()
                                next_rsc = net.get_resource(rsc_predictors)
                                throughput_results.append(next_thr)
                                resource_results.append(next_rsc)
                                l_rsc = net.get_resource(rsc_predictors, breakdown=True)
                                brakdown_results.append(l_rsc)
                                config_results.append(copy.deepcopy(config_copy))
                                if next_thr < curr_thr:
                                    break
                                else:
                                    config = config_copy
                    thr = curr_thr
                    net.update(config, 1)
                    assert curr_thr == net.get_throughput()
                    curr_rsc = net.get_resource(rsc_predictors)
                    throughput_results.append(curr_thr)
                    resource_results.append(curr_rsc)
                    config_results.append(copy.deepcopy(config))
                    l_rsc = net.get_resource(rsc_predictors, breakdown=True)
                    brakdown_results.append(l_rsc)
    config_strs = [str(c) for c in config_results]
    indices_of_unique = list(dict(map(reversed, enumerate(config_strs))).values())
    throughput_results = [throughput_results[i] for i in indices_of_unique]
    resource_results = [resource_results[i] for i in indices_of_unique]
    config_results = [config_results[i] for i in indices_of_unique]
    brakdown_results = [brakdown_results[i] for i in indices_of_unique]
    resource_optimal, throughput_optimal, breakdown_optimal = pareto_frontier(
        resource_results, throughput_results, brakdown_results
    )

    # np.savetxt(test_name+"_all.csv", np.array([throughput_results, resource_results]).T, delimiter=",")
    # np.savetxt(test_name+"_optimal.csv", np.array([throughput_optimal, resource_optimal]).T, delimiter=",")
    # with open(test_name+"_all_config.csv", 'w') as f:
    #    write = csv.writer(f)
    #    write.writerows(config_results)
    with open(test_name + "_optimal_breakdown.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(breakdown_optimal)

    plt.figure()
    plt.scatter(resource_results, throughput_results, marker="x")
    plt.scatter(resource_optimal, throughput_optimal, marker="x")
    plt.xlabel(xlabel="LUT")
    plt.ylabel(ylabel="Throughput")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(test_name)
    plt.savefig(test_name + ".png")


# if __name__ == "__main__":
#     #main("OPT125m")
#     #main("OPT350m")
#     #main("OPT1_3b")
#     main("BertSmall")
