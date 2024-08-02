from setuptools import setup, find_packages
import sys


def is_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except:
        print(
            "You need to install PyTorch before installing MASE: https://pytorch.org/get-started/locally/."
        )
        return False


def get_system():
    return sys.platform


requirements = [
    "torch",
    "torchvision",
    "onnx",
    "black",
    "toml",
    "GitPython",
    "colorlog",
    "cocotb[bus]==1.8.0",
    "pytest",
    "pytorch-lightning",
    "transformers",
    "toml",
    "timm",
    "pytorch-nlp",
    "datasets",
    "ipython",
    "ipdb",
    "sentencepiece",
    "einops",
    "pybind11",
    "tabulate",
    "tensorboardx",
    "hyperopt",
    "accelerate",
    "optuna",
    "stable-baselines3[extra]",
    "h5py",
    "scikit-learn",
    "scipy",
    "onnxruntime",
    "matplotlib",
    "sphinx-rtd-theme",
    "sphinx-test-reports",
    "sphinxcontrib-plantuml",
    "sphinx-needs",
    "imageio",
    "imageio-ffmpeg",
    "opencv-python",
    "kornia",
    "einops",
    "ghp-import",
    "optimum",
    "pytest-profiling",
    "myst_parser",
    "pytest-cov",
    "pytest-xdist",
    "pytest-sugar",
    "pytest-html",
    "lightning",
    "wandb",
    "bitarray",
    "bitstring",
    "emoji",
    "numpy<2.0",
    "tensorboard",
    "sphinx_needs",
    "onnxconverter-common",
    "absl-py",
    "sphinx-glpi-theme",
    "prettytable",
    "pyyaml",
    "pynvml",
    "bitstring>=4.2",
    "myst_parser",
    "cvxpy",
    "py-cpuinfo",
    "pandas",
    "psutil",
    "Pillow==10.4.0",
    "mpmath==1.3.0",
]

setup(
    name="mase-tools",
    version="1.0.0",
    description="Machine-Learning Accelerator System Exploration Tools",
    author="Aaron Zhao, Jianyi Cheng, Cheng Zhang, Pedro Gimenes",
    author_email="a.zhao@imperial.ac.uk, jianyi.cheng17@imperial.ac.uk, chengzhang98@outlook.com, pedro.gimenes19@imperial.ac.uk",
    license_files=("LICENSE",),
    python_requires=">=3.11.9",
    package_dir={
        "": "src",
    },
    packages=find_packages("src"),
    install_requires=requirements,
)
