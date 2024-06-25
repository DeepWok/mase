# Getting Started using Conda

## Install Conda (for the first time)

If you don't have `conda` installed yet, fetch the download link from [this page](https://www.anaconda.com/download#downloads), download with wget and execute with all default settings. For example:
```bash
# Update the download link with the appropriate version and platform
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

3. Install required dependencies:
```shell
conda env create -f machop/environment.yml
conda activate mase
pip install -r machop/requirements.txt
```

> **Common error:** when installing the `pip` requirements, make sure the conda environment is activated and the pip command points to your environment version. You can check this by running `which pip` or `which python` and making sure the resulting path includes "anaconda" and "mase".

## Install MASE

1. After all dependencies are installed, you can do an editable pip install. This makes the mase packages available within any python session in your environment.

```bash
pip install -e . -vvv
```

2. (OPTIONAL) You can check the installation was successful by running the following, which ensures MASE's software stack is available globally in your python environment.

```bash
python -c "import chop"
```

3. (OPTIONAL) You can also run the Machop test stack to ensure the codebase is running correctly by running the following command.
```bash
pytest machop/test
```