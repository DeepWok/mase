# Environment Setup Instructions

## Create and Activate Virtual Environment

```sh
conda create -n <your_venv_name> python=3.11.11 ipykernel jupyter_client pytorch
conda activate <your_venv_name>
```

## Clone and Install `mase-DLG`

```sh
cd <whereever_you_like>
git clone https://github.com/johanjino/mase-DLG
cd mase-DLG
python3 -m pip install -e . -vvv
cd ..
```

## Clone and Install `traceable-difflogic`

Note: for easier setup we are using cpu version of difflogic for now.

```sh
git clone -b main-cpu https://github.com/johanjino/traceable-difflogic
cd traceable-difflogic
python3 -m pip install -e . -vvv
cd ..
```

## Additional Requirement for Saving `masegraph`

If `pydot` is not already installed, install it using:

```sh
pip install pydot
```

