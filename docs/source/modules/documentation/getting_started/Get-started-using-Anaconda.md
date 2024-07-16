# Getting Started using Conda

## Install Conda (for the first time)

If you don't have `conda` installed yet, fetch the download link from [this page](https://www.anaconda.com/download#downloads), download with wget and execute with all default settings. For example:
```bash
# Update the download link with the appropriate version and platform
# this is the same process as the official doc, so you can also follow the official doc for installing conda
wget https://repo.anaconda.com/archive/Anaconda3-<VERSION>-<PLATFORM>.sh
chmod +x Anaconda3-<VERSION>-<PLATFORM>.sh
./Anaconda3-<VERSION>-<PLATFORM>.sh -b
```

## Install environment using Conda

1. Clone the MASE repository:
```shell
git clone git@github.com:DeepWok/mase.git
```

2. Create your own branch to work on:
```shell
cd mase
git checkout -b your_branch_name
```

3. Install MASE and its dependencies:
```shell
# we suggest you to create a new conda environment for MASE, you can do this by running the following command
# this creates separation for safe development and testing
conda create -n mase python=3.11
conda activate mase
# checking , make sure you are using the correct python version and also the correct environment
which python
python --version
# install in editable mode, this means your changes will be reflected in the package
# we suggest you use -m install instead of plain pip install to avoid any misuse
python3 -m pip install -e . -vvv
```

> **Common error:** when installing the `pip` requirements, make sure the conda environment is activated and the pip command points to your environment version. You can check this by running `which pip` or `which python` and making sure the resulting path includes "anaconda" and "mase".

## Install MASE

1. After all dependencies are installed, you can do an editable pip install. This makes the mase packages available within any python session in your environment, this command was included in the previous step. You have to understand what is meant by `editable` mode. This means that the package is installed in a way that any changes you make to the source code will be reflected in the package. This is useful for development and testing purposes. For stable installations, you can remove the `-e` flag.

```bash
python3 -m pip install -e . -vvv
```

2. (Optional but suggested) You can check the installation was successful by running the following, which ensures MASE's software stack is available globally in your python environment.

```bash
python -c "import chop"
```

3. (Optional but suggested) You can also run the Machop test stack to ensure the codebase is running correctly by running the following command.

```bash
pytest machop/test
```