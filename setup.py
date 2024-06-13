from setuptools import setup, find_packages


def is_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except:
        return False


requirements = [
    "torch",
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
    "onnxruntime>=1.16",
    "onnxruntime-tools",
    "onnxconverter_common",
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
    "numpy",
    "absl-py",
    "scipy",
    "sphinx-glpi-theme",
    "prettytable",
    "pyyaml",
    "pynvml",
    "bitstring>=4.2",
]

setup(
    name="mase-tools",
    version="1.0.0",
    description="Machine-Learning Accelerator System Exploration Tools",
    author="Aaron Zhao, Jianyi Cheng, Cheng Zhang, Pedro Gimenes",
    author_email="a.zhao@imperial.ac.uk, jianyi.cheng17@imperial.ac.uk, chengzhang98@outlook.com, pedro.gimenes19@imperial.ac.uk",
    license_files=("LICENSE",),
    python_requires=">=3.10.6",
    package_dir={
        "": "src",
    },
    packages=find_packages("src"),
    install_requires=requirements,
)
