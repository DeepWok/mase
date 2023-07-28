#!/bin/bash
# This is an installation script for Ubuntu 20.04
# Some of paths are specific for the cas server

set -e
set -x

# Install basic dependences 
sudo apt-get update -y 
sudo apt-get install -y clang cmake graphviz-dev libclang-dev \
                        pkg-config g++ libxtst6 xdg-utils \
                        libboost-all-dev llvm gcc ninja-build \
                        python3 python3-pip build-essential \
                        libssl-dev git vim wget htop sudo \
                        lld parallel clang-format clang-tidy \
                        libtinfo5 gcc-multilib libidn11-dev

# Install SystemVerilog formatter
if [ ! -d "/workspace/srcPkgs/verible" ] 
then
  (mkdir -p /workspace/srcPkgs \
  && cd /workspace/srcPkgs \
  && wget https://github.com/chipsalliance/verible/releases/download/v0.0-2776-gbaf0efe9/verible-v0.0-2776-gbaf0efe9-Ubuntu-22.04-jammy-x86_64.tar.gz \
  && mkdir -p verible \
  && tar xzvf verible-*-x86_64.tar.gz -C verible --strip-components 1)
else
  echo "verible already exists"
fi


# Install verilator from source - version v5.006 
sudo apt-get update -y 
sudo apt-get install -y git perl make autoconf flex bison \
                        ccache libgoogle-perftools-dev numactl \
                        perl-doc libfl2 libfl-dev zlib1g zlib1g-dev \
                        help2man

# Install verilator from source - version v5.006
if [ ! -d "/workspace/srcPkgs/verilator" ] 
then
  (mkdir -p /workspace/srcPkgs \
  && cd /workspace/srcPkgs \
  && git clone https://github.com/verilator/verilator)
else
  echo "verilator already exists"
fi
if [ "/usr/local/bin/verilator" ] 
then
  (mkdir -p /workspace/srcPkgs \
  && cd /workspace/srcPkgs \
  && unset VERILATOR_ROOT \
  && cd verilator \
  && git checkout v5.006 \
  && autoconf \
  && ./configure \
  && make \
  && sudo make install)
else
  echo "verilator already installed"
fi

export PATH="${PATH}:/home/vagrant/.local/bin"
pip3 install -r /workspace/machop/requirements.txt
pip3 install --pre torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    && pip3 install onnx black toml GitPython colorlog cocotb[bus]==1.8.0 \
                    pytest pytorch-lightning transformers toml \
                    timm pytorch-nlp datasets ipython ipdb \
                    sentencepiece einops deepspeed pybind11 \
                    tabulate tensorboardx hyperopt accelerate \
                    optuna stable-baselines3 


# Install Torch-MLIR and Pytorch
mkdir -p /workspace/srcPkgs \
    && cd /workspace/srcPkgs \
    && wget https://github.com/llvm/torch-mlir/releases/download/snapshot-20230525.849/torch-2.1.0.dev20230523+cpu-cp310-cp310-linux_x86_64.whl \
    && wget https://github.com/llvm/torch-mlir/releases/download/snapshot-20230525.849/torch_mlir-20230525.849-cp310-cp310-linux_x86_64.whl \
    && pip3 install torch-*.whl torch_mlir-*.whl

export VHLS="/scratch/shared/Xilinx"
# Env
if ! grep -q "Mase env" /home/vagrant/.bashrc; then
printf "\
\n# Mase env
\nexport LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LIBRARY_PATH \
\n# Basic PATH setup \
\nexport PATH=/workspace/scripts:/workspace/hls/build/bin:/home/vagrant/.local/bin:/workspace/llvm/build/bin:\$PATH:/workspace/srcPkgs/verible/bin \
\n# Vitis HLS setup \
\nexport VHLS=${VHLS} \
\n# source ${VHLS}/Vitis_HLS/2023.1/settings64.sh \
\n# MLIR-AIE PATH setup \
\nexport PATH=/workspace/mlir-aie/install/bin:/workspace/mlir-air/install/bin:\$PATH \
\nexport PYTHONPATH=/workspace/mlir-aie/install/python:/workspace/mlir-air/install/python:\$PYTHONPATH \
\nexport LD_LIBRARY_PATH=/workspace/mlir-aie/lib:/workspace/mlir-air/lib:/opt/xaiengine:\$LD_LIBRARY_PATH \
\n# Thread setup \
\nexport nproc=\$(grep -c ^processor /proc/cpuinfo) \
\n# Terminal color... \
\nexport PS1=\"[\\\\\\[\$(tput setaf 3)\\\\\\]\\\t\\\\\\[\$(tput setaf 2)\\\\\\] \\\u\\\\\\[\$(tput sgr0)\\\\\\]@\\\\\\[\$(tput setaf 2)\\\\\\]\\\h \\\\\\[\$(tput setaf 7)\\\\\\]\\\w \\\\\\[\$(tput sgr0)\\\\\\]] \\\\\\[\$(tput setaf 6)\\\\\\]$ \\\\\\[\$(tput sgr0)\\\\\\]\" \
\nexport LS_COLORS='rs=0:di=01;96:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01' \
\nalias ls='ls --color' \
\nalias grep='grep --color'\n" >> /home/vagrant/.bashrc
#Add vim environment
printf "\
\nset autoread \
\nautocmd BufWritePost *.cpp silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.c   silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.h   silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.hpp silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.cc  silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.py  silent! set tabstop=4 shiftwidth=4 expandtab \
\nautocmd BufWritePost *.py  silent! !python3 -m black <afile> \
\nautocmd BufWritePost *.sv  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost *.v  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost * redraw! \
\n" >> /home/vagrant/.vimrc
fi

# Mase env
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export PATH=/workspace/scripts:/workspace/hls/build/bin:/home/vagrant/.local/bin:/workspace/llvm/build/bin:$PATH:/workspace/srcPkgs/verible/bin
# Thread setup
export nproc=$(grep -c ^processor /proc/cpuinfo)
# Terminal color...
export PS1="[\\[$(tput setaf 3)\\]\t\\[$(tput setaf 2)\\] \u\\[$(tput sgr0)\\]@\\[$(tput setaf 2)\\]\h \\[$(tput setaf 7)\\]\w \\[$(tput sgr0)\\]] \\[$(tput setaf 6)\\]$ \\[$(tput sgr0)\\]"
export LS_COLORS='rs=0:di=01;96:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01'
