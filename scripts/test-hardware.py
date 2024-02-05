#! /usr/bin/env python3
# ---------------------------------------
# This script runs the hardware regression test
# ---------------------------------------
from argparse import ArgumentParser
import sys, os, time, logging, colorlog, glob, subprocess, multiprocessing, shutil, functools
from multiprocessing import Process, Queue
import maselogger

# add components to path
from mase_components.deps import MASE_HW_DEPS  # TODO: check deps again


# ---------- TestHardware class --------------
class TestHardware:
    def __init__(self, args):
        self.args = args
        self.isdebug = self.args.debug
        # Root path of mase-tools
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.logger = maselogger.getLogger(
            "test-hardware", os.path.join(self.root, "test-hardware.log")
        )

        # Initialize test cases
        self.test_cases = self.args.test_cases
        if self.args.run_all:
            self.test_cases = list(MASE_HW_DEPS.keys())
        self.logger.debug(f"Test cases: ")
        self.logger.debug("\n           ".join(self.test_cases))

        check_fail = self.check_files()
        if check_fail:
            sys.exit(check_fail)

    def check_files(self):
        """Check if all the required files exist for test cases"""
        for test_case in self.test_cases:
            group, module = test_case.split("/")
            # Check if the test bench exists
            tb_path = os.path.join(
                self.root, "machop/mase_components", group, f"test/{module}_tb.py"
            )
            if not os.path.exists(tb_path):
                self.logger.error(
                    f"Cannot find the testbench! Expected to be {tb_path}"
                )
                return 1
            # Check if the design file exists
            rtl_file = os.path.join(
                self.root, f"machop/mase_components/{group}/rtl/{module}.sv"
            )
            if not os.path.isfile(rtl_file):
                self.logger.error(
                    f"Cannot find the design file! Expected to be {rtl_file}"
                )
                return 1
        return 0

    def lint(self):
        """Lint all the required design files"""
        cwd = os.path.join(self.root)  # What is this for?
        self.logger.info("Linting all the design files...")
        for test_case in self.test_cases:
            group, module = test_case.split("/")
            rtl_file = os.path.join(
                self.root, "machop/mase_components", group, f"rtl/{module}.sv"
            )
            include_files = [
                f"-I{os.path.join(self.root, 'machop/mase_components', group, 'rtl')}"
                for group in MASE_HW_DEPS[test_case]
            ]
            cmd = [
                "verilator",
                "--lint-only",
                "--Wall",
                # These errors are in later versions of verilator
                "-Wno-GENUNNAMED",  # Too many existing errors
                "-Wno-WIDTHEXPAND",
                "-Wno-WIDTHTRUNC",
                rtl_file,
            ] + include_files
            result, _ = self.execute(cmd, log_output=self.isdebug, cwd=cwd)
            if result:
                return 1
        return 0

    def test(self):
        """Test the given test cases"""
        test_count = len(self.test_cases)

        jobs = [None] * test_count
        queue = Queue(test_count)
        for i, test_case in enumerate(self.test_cases):
            group, module = test_case.split("/")
            jobs[i] = Process(target=self.single_test, args=(group, module, queue))
            jobs[i].start()

        for job in jobs:
            job.join()

        err = 0
        for _ in range(test_count):
            err += queue.get()
        if err:
            self.logger.error(
                "Hardware regression test finished. {} errors.".format(err)
            )
        else:
            self.logger.info(
                "Hardware regression test finished. {} errors.".format(err)
            )

        cwd = os.path.join(self.root, "machop/mase_components", "testbench")
        pycache_files = os.path.join(cwd, "__pycache__")
        if os.path.exists(pycache_files):
            shutil.rmtree(pycache_files)
        return err

    def single_test(self, group, module, queue):
        self.logger.info(f"Running unit test for {group}/{module}...")
        cwd = os.path.join(self.root, "machop/mase_components", group, "test")
        cmd = ["python3", "{}_tb.py".format(module)]

        result, buff = self.execute(cmd, log_output=self.isdebug, cwd=cwd)
        # Cocotb returns 0 even when the result is wrong. Here check log file as well
        if result or "FAIL=0" not in buff:
            self.logger.error(f"FAIL: {group}/{module}.")
            result = 1
        else:
            # Clean files
            pycache_files = os.path.join(cwd, "__pycache__")
            simbuild_files = os.path.join(cwd, "sim_build")
            result_file = os.path.join(cwd, "results.xml")
            if os.path.exists(pycache_files):
                shutil.rmtree(pycache_files)
            if os.path.exists(simbuild_files):
                shutil.rmtree(simbuild_files)
            if os.path.isfile(result_file):
                os.remove(result_file)
        queue.put(result)
        return result

    def execute(self, cmd, log_output: bool = True, log_file=None, cwd="."):
        buff = ""
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
                    buff += line
                    if log_file and log_output:
                        f.write(line)
                    line = line.rstrip("\n")
                    if log_output:
                        self.logger.info(line)
            if result.stderr:
                for line in result.stderr:
                    buff += line
                    if log_file and log_output:
                        f.write(line)
                    line = line.rstrip("\n")
                    if log_output:
                        self.logger.info(line)
            if log_file:
                f.close()
        return result.returncode, buff


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
        help="Run the whole regression test, Default=False",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="Run in debug mode, Default=False",
    )
    parser.add_argument(
        "-t",
        "--test",
        default="",
        nargs="+",
        dest="test_cases",
        help="Test individual cases",
    )

    args = parser.parse_args()
    testrun = TestHardware(args)
    lint = testrun.lint()
    if lint:
        sys.exit(lint)
    run = testrun.test()
    if run:
        sys.exit(run)
    sys.exit(0)


if __name__ == "__main__":
    main()
