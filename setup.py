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
    "torch==2.6",
    "torchvision",
    "onnx",
    "black",
    "toml",
    "GitPython",
    "colorlog",
    "cocotb==1.9.2",
    "pytest",
    "pytorch-lightning",
    "transformers==4.49",
    "toml",
    "timm",
    "pytorch-nlp",
    "datasets==3.3.2",
    "evaluate==0.4.3",
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
    "scipy==1.14.1",
    "onnxruntime",
    "matplotlib",
    "sphinx-rtd-theme",
    "sphinx-needs>=4",
    "sphinx-test-reports @ git+https://github.com/useblocks/sphinx-test-reports",
    "sphinxcontrib-plantuml",
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
    "myst-nb",
    "sphinx-book-theme",
    "pydot",
    "attr-dot-dict",
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
