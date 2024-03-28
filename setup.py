from setuptools import setup, find_packages
import subprocess
import os

setup(
    name="mase-tools",
    version="1.0.0",
    description="Machine-Learning Accelerator System Exploration Tools",
    author="Aaron Zhao, Jianyi Cheng, Cheng Zhang, Pedro Gimenes",
    author_email="a.zhao@imperial.ac.uk, jianyi.cheng17@imperial.ac.uk, chengzhang98@outlook.com, pedro.gimenes19@imperial.ac.uk",
    license_files=("LICENSE",),
    python_requires=">=3.10.6",
    package_dir={
        "": "machop",
    },
    packages=find_packages("machop"),
    install_requires=[
        "torch",
        "pythran",
        "torchvision",
        "torchaudio",
        "packaging",
        "py-cpuinfo",
        "psutil",
        "lightning",
        "transformers",
        "diffusers",
        "accelerate",
        "toml",
        "timm",
        "pytorch-nlp",
        "datasets",
        "onnx",
        "onnxruntime",
        "optimum",
        "black",
        "GitPython",
        "colorlog",
        "cocotb[bus]==1.8.0",
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "pytest-sugar",
        "pytest-html",
        "pytest-profiling",
        "ipython",
        "ipdb",
        "sentencepiece",
        "einops",
        "deepspeed",
        "pybind11",
        "tabulate",
        "tensorboard",
        "optuna",
        "stable-baselines3",
        "scikit-learn",
        "h5py",
        "pyyaml",
        "numpy",
        "pandas",
        "wandb",
        "imageio",
        "imageio-ffmpeg",
        "opencv-python",
        "kornia",
        "einops",
        "sphinx",
        "sphinx-rtd-theme",
    ],
)
