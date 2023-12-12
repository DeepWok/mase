# For hardware project students: Get started on local machines

This document includes the steps to install Mase on your local machine.

## Operating system requirements

This document targets Unix system users. If you use Windows, you will need to install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) first and then follow the following instructions in the Linux subsystem.

## Prerequisites

It is highly recommended to install [Docker](https://www.docker.com/), otherwise you will need to manually install all the dependences.

## Install Mase

1. Clone Mase to your directory:
```shell
cd ${HOME}
git clone git@github.com:JianyiCheng/mase-tools.git
```

2. Create your own branch to work on:
```shell
cd ${HOME}/mase-tools
git checkout -b your_branch_name
```

3. Build the Docker container and install Mase:
```shell
cd ${HOME}/mase-tools
make shell
```
Then you will enter the Docker container under a directory named `workspace`. This is the main directory of the `mase-tools`. You can build your work from there.

## Use Mase

If you are running a new terminal after installation:
```shell
cd /home/${USERNAME}/mase-tools
make shell
```

### Quick Test

Now let's test with a small example `common/int_mult`. The RTL code is named `hardware/common/int_mult.sv` and the test bench is named `hardware/testbench/hardware/common/int_mult.sv'. To run the test bench:
```shell
cd /workspace
make test-hw
```

