#! /usr/bin/env python3
# ---------------------------------------
# This script runs the regression test
# ---------------------------------------
import functools
import glob
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import colorlog
import maselogger

software_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "software")
)

# {model}, {task}, {dataset}, {toml}
# ./chop --modify-sw-config {toml} --dataset {dataset} --model {model} --task {task} --debug  --project={model} --synthesize auto

simple_cases = [
    [
        "toy",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "resnet18",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "mobilenetv3_small",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "efficientnet_v2_s",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "pvt_v2_b0",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "deit_tiny_224",
        "cls",
        "imagenet",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-125m@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-350m@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
]

full_cases = [
    [
        "resnet50",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "mobilenetv3_large",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "efficientnet_v2_m",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "efficientnet_v2_l",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "pvt_v2_b1",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "pvt_v2_b2",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "pvt_v2_b3",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "pvt_v2_b4",
        "cls",
        "cifar10",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "deit_small_224",
        "cls",
        "imagenet",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "deit_base_224",
        "cls",
        "imagenet",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-1.3b@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-2.7b@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-13b@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-30b@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    [
        "facebook/opt-66b@patched",
        "cls",
        "boolq",
        "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml")),
    ],
    # ["bert-base-uncased", "cls", "boolq", "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml"))],
    # ["bert-base-cased", "cls", "boolq", "{}".format(os.path.join(software_dir, "configs", "modify-sw", "integer.toml"))],
] + simple_cases


# ---------- TestHardware class --------------
class TestMicro:
    def __init__(self, args):
        self.args = args
        self.to_debug = args.debug
        # Root path of mase-tools
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.logger = maselogger.getLogger(
            "test-micro", os.path.join(self.root, "test-micro.log")
        )

        self.project_dir = (
            os.path.join(self.root, "mase_output")
            if args.project_dir is None
            else args.project_dir
        )
        if not os.path.exists(self.project_dir):
            os.mkdir(self.project_dir)
        if args.reset:
            for p in glob.glob(os.path.join(self.project_dir, "*")):
                if os.path.isfile(p):
                    os.remove(p)
                else:
                    shutil.rmtree(p)

        self.mode = args.mode
        assert self.mode in ["auto", "hls"], f"Invalid synthesizing mode: {self.mode}"

        self.test_cases = []
        if self.args.run_all:
            self.test_cases = full_cases
        elif self.args.run_simple:
            self.test_cases = simple_cases
        for test in args.test_cases:
            assert test in [
                f"{x[0]}_{x[1]}_{x[2]}" for x in full_cases
            ], f"Cannot find test case: {test}"
            for x in full_cases:
                if test == f"{x[0]}_{x[1]}_{x[2]}" and test not in [
                    f"{x[0]}_{x[1]}_{x[2]}" for x in self.test_cases
                ]:
                    self.test_cases.append(x)

    def test(self, parallel=False):
        """Test the given test cases"""
        if parallel:
            test_count = len(self.test_cases)
            jobs = [None] * test_count
            queue = Queue(test_count)
            for i, test_case in enumerate(self.test_cases):
                jobs[i] = Process(target=self.single_test, args=(test_case, queue))
                jobs[i].start()

            for job in jobs:
                job.join()

            err = 0
            for _ in range(test_count):
                err += queue.get()
        else:
            test_count = len(self.test_cases)
            err = 0
            queue = Queue(test_count)
            for i, test_case in enumerate(self.test_cases):
                err += self.single_test(test_case, queue) > 0

        if err:
            self.logger.error("Regression test finished. {} errors.".format(err))
        else:
            self.logger.info("Regression test finished. {} errors.".format(err))
        return err

    def single_test(self, test_case, queue):
        self.logger.info("Running unit test for {}...".format(test_case))

        model = test_case[0]
        task = test_case[1]
        dataset = test_case[2]
        sw_modify_toml = test_case[3]

        result = 0
        cmd = [
            f"./chop",
            f"--modify-sw-config",
            f"{sw_modify_toml}",
            f"--model",
            f"{model}",
            f"--task",
            f"{task}",
            f"--dataset",
            f"{dataset}",
            f"--debug",
            f"--synthesize",
            f"{self.args.mode}",
            f"--project",
            f"{model}_{task}_{dataset}",
            f"--project-dir",
            f"{self.project_dir}",
        ]
        result = self.execute(
            cmd, log_output=self.to_debug, cwd=os.path.join(self.root, "software")
        )

        queue.put(result)
        return result

    def execute(self, cmd, log_output: bool = True, log_file=None, cwd="."):
        if log_output:
            self.logger.debug(subprocess.list2cmdline(cmd))
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=cwd
            ) as result:
                if log_file:
                    f = open(log_file, "w")
                if result.stdout or result.stderr:
                    self.logger.info("")
                if result.stdout:
                    for line in result.stdout:
                        if log_file:
                            f.write(line)
                        line = line.rstrip("\n")
                        self.logger.trace(line)
                if result.stderr:
                    for line in result.stderr:
                        if log_file:
                            f.write(line)
                        line = line.rstrip("\n")
                        self.logger.trace(line)
                if log_file:
                    f.close()
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=cwd)
        return result.returncode


# ---------- main function --------------
def main():
    USAGE = """Usage:
test-hardware.py --test common/int_mult ...
test-hardware.py -a"""

    parser = ArgumentParser(usage=USAGE)
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="run_all",
        default=False,
        help="Run the regression test for the full cases. Default=False",
    )
    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        dest="run_simple",
        default=False,
        help="Run the regression test for simple cases. Default=False",
    )
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        dest="reset",
        default=False,
        help="Clean the project directory. Default=False",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        default=None,
        help="Output dir. Default=$MASE/mase_output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="Run in debug mode. Default=False",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="hls",
        dest="mode",
        help="Test synthesize mode. Default=hls",
    )
    parser.add_argument(
        "-t",
        "--test",
        default="",
        nargs="+",
        dest="test_cases",
        help="Test individual cases. Format = {model}_{task}_{dataset}",
    )
    parser.add_argument(
        "-j",
        default=False,
        action="store_true",
        dest="parallel",
        help="Enable parallel testing. Default = False",
    )

    args = parser.parse_args()
    testrun = TestMicro(args)
    run = testrun.test(parallel=args.parallel)
    if run:
        sys.exit(run)
    sys.exit(0)


if __name__ == "__main__":
    main()
