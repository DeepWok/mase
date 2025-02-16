#! /usr/bin/env python3
# ---------------------------------------
# This script calls the mase hls tool
# ---------------------------------------
from argparse import ArgumentParser
import sys, os, time, logging, colorlog, glob, subprocess, multiprocessing, shutil, functools
from multiprocessing import Process, Queue
import maselogger


# ---------- TestHardware class --------------
class MaseHLS:
    def __init__(self, args):
        self.args = args
        self.isdebug = self.args.debug
        # Root path of mase-tools
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.logger = maselogger.getLogger(
            "mase-hls", os.path.join(self.root, "mase-hls.log")
        )
        self.run()

    def run(self):
        result = 1
        if self.args.build_all or self.args.build:
            result |= self.build()
        if self.args.quick_test:
            result |= self.quick_test()

    def build(self):
        if self.args.build_all:
            cmd = [
                "make",
                "build",
            ]
            return self.execute(cmd, log_output=True, cwd=self.root)
        elif self.args.build:
            cmd = [
                "bash",
                os.path.join("scripts", "build-mase-hls.sh"),
            ]
            return self.execute(cmd, log_output=True, cwd=self.root)
        return True

    def quick_test(self):
        shutil.copy(
            os.path.join(
                self.root,
                "test",
                "test_in.mlir",
            ),
            os.path.join(
                self.root,
                "test",
                "test.mlir",
            ),
        )
        result = False
        cmd = [
            "mase-opt",
            "--preprocess-func=func-name=relu",
            "--canonicalize",
            os.path.join(
                self.root,
                "test",
                "test.mlir",
            ),
            "-o",
            os.path.join(
                self.root,
                "test",
                "test1.mlir",
            ),
        ]
        result |= self.execute(cmd, log_output=True, cwd=self.root)

        cmd = [
            "mase-opt",
            "--emit-hls",
            #             "--emit-hls=file-name="
            #             + os.path.join(
            #                 self.root,
            #                 "test",
            #                 "test.cpp",
            #             ),
            os.path.join(
                self.root,
                "test",
                "test1.mlir",
            ),
            "--debug",
        ]
        result |= self.execute(cmd, log_output=True, cwd=self.root)

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
mase-hls.py -b
"""

    parser = ArgumentParser(usage=USAGE)
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        dest="build",
        default=False,
        help="Build the mase hls tool, Default=False",
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        dest="build_all",
        default=False,
        help="Build the mase hls tool from scratch (including LLVM), Default=False",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="Run in debug mode, Default=False",
    )
    # parser.add_argument(
    #     "-t",
    #     "--test",
    #     default="",
    #     nargs="+",
    #     dest="test_cases",
    #     help="Test individual cases",
    # )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        default=False,
        dest="quick_test",
        help="Test individual cases",
    )

    args = parser.parse_args()
    MaseHLS(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
