Before you start, please make sure you have `conda` installed to ensure a easy manipulation of `Python` packages and environments.

If you do not have `conda`, please check advice on the documentation in [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html). All the guide and installations have only be tested on unix-like OS.

The following lines should print out your current python version

* Install Anaconda on your Linux/WSL2/Mac.

* Clone the mase-tools to your pc.

 ```bash
 git clone git@github.com:JianyiCheng/mase-tools.git
 ```

* Run the `int-conda.sh` script to create a Conda environment named "mase" and install related packages:

 ```bash
 cd mase-tools
 bash scripts/init-conda.sh
 ```

* Activate `mase` env when using mase-tools

 ```bash
 conda activate mase
 ```

* Create your own branch to work on:

 ```bash
 git checkout -b your_branch_name
 ```
