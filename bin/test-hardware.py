#! /usr/bin/env python3
# ---------------------------------------
# This script runs the hardware regression test
# ---------------------------------------
from optparse import OptionParser
import sys, os, time, logging, colorlog, glob, subprocess, multiprocessing, shutil, functools
import maselogger

# Add more test cases here
testcases = {'common/int_mult'}


# ---------- testhardware class --------------
class testhardware:

    def __init__(self, args, options):
        self.args = args
        self.options = options
        self.isdebug = self.options.debug
        # Root path of mase-tools
        self.root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        self.logger = maselogger.getLogger(
            'test-hardware', os.path.join(self.root, 'test-hardware.log'))
        self.testcases = self.args
        if self.options.runall: self.testcases = testcases
        checkfiles = self.checkfiles()
        if checkfiles:
            sys.exit(checkfiles)

    def checkfiles(self):
        """Check if all the required files exist for test cases"""
        for testcase in self.testcases:
            basename = os.path.basename(testcase)
            # Check if the test bench exists
            if not os.path.exists(
                    os.path.join(self.root, 'hardware', 'testbench', testcase,
                                 '{}_tb.py'.format(basename))):
                self.logger.error(
                    'Cannot find the testbench! Expected to be {}'.format(
                        os.path.join(self.root, 'hardware', 'testbench',
                                     testcase, '{}_tb.py'.format(basename))))
                return 1
            # Check if the design file exists
            if not os.path.exists(
                    os.path.join(self.root, 'hardware',
                                 '{}.sv'.format(testcase))):
                self.logger.error(
                    'Cannot find the design file! Expected to be {}'.format(
                        os.path.join(self.root, 'hardware',
                                     '{}.sv'.format(testcase))))
                return 1
        return 0

    def lint(self):
        """Lint all the required design files"""
        cwd = os.path.join(self.root)
        self.logger.info('Linting all the design files...')
        for testcase in self.testcases:
            designfile = os.path.join(self.root, 'hardware', testcase)
            cmd = [
                'verilator', '--lint-only', '--Wall',
                '{}.sv'.format(designfile)
            ]
            result = self.execute(cmd, logoutput=self.isdebug, cwd=cwd)
            if result:
                return 1
        return 0

    def test(self):
        """Test the given test cases"""
        err = 0
        for testcase in self.testcases:
            self.logger.info('Running unit test for {}...'.format(testcase))
            basename = os.path.basename(testcase)
            cwd = os.path.join(self.root, 'hardware', 'testbench', testcase)
            cmd = ['python3', '{}_tb.py'.format(basename)]
            result = self.execute(cmd, logoutput=self.isdebug, cwd=cwd)
            if result:
                self.logger.error('FAIL.')
            else:
                # Clean files
                pycachefiles = os.path.join(cwd, '__pycache__')
                simbuildfiles = os.path.join(cwd, 'sim_build')
                resultfile = os.path.join(cwd, 'results.xml')
                if os.path.exists(pycachefiles):
                    shutil.rmtree(pycachefiles)
                if os.path.exists(simbuildfiles):
                    shutil.rmtree(simbuildfiles)
                if os.path.isfile(resultfile):
                    os.remove(resultfile)
            err += result

        self.logger.info(
            'Hardware regression test finished. {} errors.'.format(err))
        return err

    def runall(self):
        self.logger.info('Running hardware regression test...')
        err = 0
        for key, testcase in tblibpath.items():
            self.logger.info('Running test case for {}...'.format(key))
            result = self.rununittest(key, testcase)
            if result:
                self.logger.error('Simulation for {} failed: {}.'.format(
                    testcase, cmdline))
                self.logger.info('FAILED')
                err += 1
            else:
                self.logger.info('SUCCESS')
        return err

    def debugtests(self):
        for arg in self.args:
            self.logger.info('Running unit test for {}...'.format(arg))
            if self.debugunittest(arg, tblibpath[arg]):
                return 1
        return 0

    def rununittest(self, key, designfiles):
        basename = os.path.basename(key)
        cwd = os.path.join(self.root, 'hardware', 'testbench', key)
        obj = os.path.join(cwd, 'obj_dir')
        if os.path.exists(obj):
            shutil.rmtree(obj)
        designfileswithpath = [
            os.path.join(self.root, 'hardware', v) for v in designfiles
        ]
        cmd = [
            'verilator', '-Wall', '-cc', '--trace'
        ] + designfileswithpath + [
            '--top-module', basename, '--exe', '{}_tb.cpp'.format(basename)
        ]
        cmdline = subprocess.list2cmdline(cmd)
        result = self.execute(cmd, logoutput=False, cwd=cwd)
        if result:
            return 1
        cmd = [
            'make', '-j', '-C', 'obj_dir/', '-f', 'V{}.mk'.format(basename),
            'V{}'.format(basename)
        ]
        cmdline = subprocess.list2cmdline(cmd)
        result = self.execute(cmd, logoutput=False, cwd=cwd)
        if result:
            return 1
        cmd = ['obj_dir/V{}'.format(basename)]
        cmdline = subprocess.list2cmdline(cmd)
        result = self.execute(cmd, logoutput=False, cwd=cwd)
        if result:
            return 1
        shutil.rmtree(obj)
        return 0

    def debugunittest(self, key, designfiles):
        basename = os.path.basename(key)
        cwd = os.path.join(self.root, 'hardware', 'testbench', key)
        obj = os.path.join(cwd, 'obj_dir')
        if os.path.exists(obj):
            shutil.rmtree(obj)
        designfileswithpath = [
            os.path.join(self.root, 'hardware', v) for v in designfiles
        ]
        cmd = [
            'verilator', '-Wall', '-cc', '--trace'
        ] + designfileswithpath + [
            '--top-module', basename, '--exe', '{}_tb.cpp'.format(basename)
        ]
        cmdline = subprocess.list2cmdline(cmd)
        self.logger.debug(cmdline)
        # Emit results to the console for debugging
        result = self.execute(cmd, cwd=cwd)
        if result:
            self.logger.error('Simulation for {} failed: {}.'.format(
                key, cmdline))
            return 1
        cmd = [
            'make', '-j', '-C', 'obj_dir/', '-f', 'V{}.mk'.format(basename),
            'V{}'.format(basename)
        ]
        cmdline = subprocess.list2cmdline(cmd)
        self.logger.debug(cmdline)
        result = self.execute(cmd, cwd=cwd)
        if result:
            self.logger.error('Simulation for {} failed: {}.'.format(
                key, cmdline))
            return 1
        cmd = ['obj_dir/V{}'.format(basename)]
        cmdline = subprocess.list2cmdline(cmd)
        self.logger.debug(cmdline)
        result = self.execute(cmd, cwd=cwd)
        if result:
            self.logger.error('Simulation for {} failed: {}.'.format(
                key, cmdline))
            return 1
        shutil.rmtree(obj)
        return 0

    def execute(self, cmd, logoutput: bool = True, logfile=None, cwd='.'):
        if logoutput:
            with subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  bufsize=1,
                                  universal_newlines=True,
                                  cwd=cwd) as result:
                if logfile: f = open(logfile, 'w')
                if result.stdout or result.stderr:
                    self.logger.info('')
                if result.stdout:
                    for line in result.stdout:
                        if logfile: f.write(line)
                        line = line.rstrip("\n")
                        self.logger.trace(line)
                if result.stderr:
                    for line in result.stderr:
                        if logfile: f.write(line)
                        line = line.rstrip("\n")
                        self.logger.trace(line)
                if logfile: f.close()
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=cwd)
        return result.returncode


# ---------- main function --------------
def main():
    USAGE = """Usage: 
test-hardware.py common/int_mult ...
test-hardware.py -a"""

    optparser = OptionParser(usage=USAGE)
    optparser.add_option('-a',
                         '--all',
                         action='store_true',
                         dest='runall',
                         default=False,
                         help='Run the whole regression test, Default=False')
    optparser.add_option('--debug',
                         action='store_true',
                         dest='debug',
                         default=False,
                         help='Run in debug mode, Default=False')

    (options, args) = optparser.parse_args()
    testrun = testhardware(args, options)
    lint = testrun.lint()
    if lint: sys.exit(lint)
    run = testrun.test()
    if run: sys.exit(run)
    sys.exit(0)


if __name__ == '__main__':
    main()
