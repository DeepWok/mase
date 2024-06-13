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

4. Optional, install `verilator`

Follow the instructions on official [verilator](https://verilator.org/guide/latest/install.html) installation.


## Tested Systems

- [x] darwin aarch64 mac
- [ ] linux x86_64 cuda-enabled
- [ ] windows x86_64 cuda-enabled

## Troubleshooting

1. Verilator installation

We expect the user to self-install `verilator`, and the `verilator` install is not included in the provided `nix` shell.

2. Clang problem on darwin aarch64 systems

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