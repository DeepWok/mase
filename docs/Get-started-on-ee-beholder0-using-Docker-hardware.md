# For hardware project students: Get started on ee beholder0 (using Docker)

Mase can be directly installed and used on local machines. This document includes the steps to install Mase on the ee-server (`ee-beholder0` for this example).

## VS Code (optional)
My suggestion to work on the server is to install VS Code on your local machine and develop with it. You can also use your own IDE for the project, such as `emacs` or `vim`. 

1. Download and install [VS Code](https://code.visualstudio.com/)
2. Click the *extension* icon on the left bar (Cntrl + Shift + x), and search for *Remote - SSH* and *Remote Development* packages and install them.
3. Click *Help* at the top and choose *Show All Commands* (Cntrl + Shift + P) and open the console to run a few commands.
  - Type `ssh` in the console and choose *Remote SSH - Add New SSH Host*
  - Then you are asked to enter your ssh command, in our case, use: `ssh username@ee-beholder0.ee.ic.ac.uk`
  - Enter your password and you are in! (You may be asked to choose the platform for the first time, choose *Linux*).
4. Click *File* at the top-left and choose *Open Folder* to add the directory where you do your work to the explorer.
5. To use Terminal, you can click *Terminal* on the top and choose *New Terminal*.

Then you have a basic workspace!

## Install Mase
1. In the terminal, ssh to the server (skip this if you already log in through VS Code):

```shell
ssh ${USERNAME}@ee-beholder0.ee.ic.ac.uk
```
PS: For the first time logging in, the server has not created your home directory. Try to exit and re-login will fix the problem.

2. Clone Mase to your directory:
```shell
cd /home/${USERNAME}
git clone git@github.com:JianyiCheng/mase-tools.git
```

3. Create your own branch to work on:
```shell
cd /home/${USERNAME}/mase-tools/
git checkout -b your_branch_name
```

4. Start a Docker container as a virtual environment:
```shell
cd /home/${USERNAME}/mase-tools/
make shell
```
This may take a long time for the first time. If you are working with an unstable connection, you can try [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).
The directory named `/workspace` is the main directory of the `mase-tools`. You can build your work from there.

## Use Mase

If you are running a new terminal after installation:
```shell
cd /home/${USERNAME}/mase-tools/
make shell
```

### Quick Test

Now let's test with a small example `common/int_mult`. The RTL code is named `hardware/common/int_mult.sv` and the test bench is named `hardware/testbench/hardware/common/int_mult.sv'. To run the test bench:
```shell
cd /workspace
test-hardware.py common/int_mult
# Full hardware test
make test-hw
```

