# Getting Started using Nix (Suggested)

## Install Nix (for the first time)

If you don't have `nix` installed yet, fetch the download link from [this page](https://nixos.org/download/#nix-install-macos).

`nix` is a package manager that allows us to configure the system beyond pythonic libraries.

## Install environment using Nix

1. Clone the MASE repository:
```shell
git clone git@github.com:DeepWok/mase.git
```

2. Create your own branch to work on:
```shell
cd mase
git checkout -b your_branch_name
```

3. Activate the `nix` shell:
```shell
nix-shell
```


## Tested Systems

- [x] darwin aarch64 mac
- [ ] linux x86_64 cuda-enabled
- [x] windows x86_64 cuda-enabled

## Troubleshooting

1. Clang problem on `darwin aarch64` systems

	There is some legacy issue with porting `clang` on `nix`, or generally `nix` shells and Python packages with C++ Extensions on macOS, you can find the issue detailed [here](https://discourse.nixos.org/t/nix-shells-and-python-packages-with-c-extensions/26326).

	MacOS users should follow the standard procedure to install `xcode` , which provides you with `Apple's clang/clang++ compilation tools `. In our `setup.py`, several installation would use this local system `clang`, you should be able to verify this by typing

	```shell
	[nix-shell:~/Projects/mase]$ which clang
	# The following is the expected output
	# /usr/bin/clang
	[nix-shell:~/Projects/mase]$ clang --version
	# The following is the expected output
	# Apple clang version 15.0.0 (clang-1500.3.9.4)
	# Target: arm64-apple-darwin23.5.0
	# Thread model: posix
	# InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
	```

	Using other `clang` variants, especially the llvm-backed `nix` `clang` will cause installation or running issues with `cocotb` and `verilator` because of the confusion in `std` library paths.

2. `g++` or `glibc` problem on WSL (Ubuntu-24.04 system)

	While using `verilator` in the nix shell, some of the generated files were compiled by `g++` (or `gcc`), you might encounter issues about `g++ can not find` or `glibc-xxx cannot find`, etc.
	
	This is because the nix shell does not include any `g++` or `glibc` in the environment. When compiling files, it automatically calls the related libraries in your local environment.

	When facing this kind of problem, you need to configure the local environment to match nix shell requirement.
	For instance, install or update the related packages to the `verilator` required version would work:

	```
	(.venv) cx922@DESKTOP-UAFT8QR:~/mase$ ldd --version 
	ldd (Ubuntu GLIBC 2.39-0ubuntu8.2) 2.39
	...

3. `verilator` installation and its `cocotb` integration

	We expect users to self-install `verilator` on their local system. The `nix-shell` will not install `verilator` for you. You can find the installation guide [here](https://verilator.org/guide/latest/install.html).

	One common problem we found with using `verilator` backed `cocotb` is that the `cocotb` flow lacks resolution when your `python` is running in an advanced virtual environment. The `cocotb` flow makes mistakes in identifying certain paths. One error is something in the following form:

	```bash
	- Verilator: Built from 0.000 MB sources in 0 modules, into 0.000 MB in 0 C++ files needing 0.000 MB
	- Verilator: Walltime 0.003 s (elab=0.000, cvt=0.000, bld=0.000); cpu 0.000 s on 8 threads
	INFO: Running command make -C /Users/name/Projects/mase/src/mase_components/activations/test/build/fixed_gelu/test_0 -f Vtop.mk in directory /Users/name/Projects/mase/src/mase_components/activations/test/build/fixed_gelu/test_0
	make: *** No rule to make target `/usr/local/lib/python3.11/dist-packages/cocotb/share/lib/verilator/verilator.cpp', needed by `verilator.o'.  Stop.
	```

	This error message is basically saying that the `cocotb` flow is looking for the `verilator.cpp` file in the wrong path. The `verilator.cpp` file is actually located in the `cocotb` package, which is installed in the `dite-packages` directory of your `python` environment. The `cocotb` flow is looking for the `verilator.cpp` file in the native directory, which is incorrect. One hack for this problem is to change the check in the `cocotb` `Makefile`.

	Edit the file `/Users/yz10513/anaconda3/envs/mase/lib/python3.11/site-packages/cocotb/share/makefiles/Makefile.inc` and change the following line:

	```bash
	# Our comments: this is exactly the problem, sometimes we would like to use 	
	# this ensures we use the same python as the one cocotb was installed into
	ifeq ($(IS_VENV),True)
			# In a virtual environment, the Python binary may be a symlink, so it should not use realpath
			PYTHON_BIN ?= $(shell cocotb-config --python-bin)
	else
			# disable the use of realpath!	
			# realpath to convert windows paths to unix paths, like cygpath -u
			#PYTHON_BIN ?= $(realpath $(shell cocotb-config --python-bin))
			PYTHON_BIN ?= $(shell cocotb-config --python-bin)
	endif
	```