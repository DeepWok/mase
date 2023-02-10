#!/bin/bash
# This is an installation script for Ubuntu 20.04
# Some of paths are specific for the cas server

set -e
set -x

# Install basic dependences 
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install apt-utils -y
sudo apt-get install -y \
                clang cmake graphviz-dev libclang-dev \
                pkg-config g++ llvm libxtst6 xdg-utils \
                libboost-all-dev gcc ninja-build \
                libssl-dev git vim wget htop sudo \
                lld clang-format clang-tidy build-essential \
                perl make autoconf flex bison libunwind-dev \
                ccache libgoogle-perftools-dev numactl \
                perl-doc libfl2 libfl-dev zlib1g zlib1g-dev \
                help2man python3 python3-pip python3-setuptools

pip3 install --user --upgrade pip
pip3 install onnx yapf toml GitPython colorlog cocotb[bus] pytest setuptools

# Install PyTorch and Torch-MLIR
pip3 install --pre torch-mlir torchvision \
-f https://llvm.github.io/torch-mlir/package-index/ \
--extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install SystemVerilog formatter
if [ ! -d "/home/vagrant/srcPkgs/verible" ] 
then
  (mkdir -p /home/vagrant/srcPkgs \
  && cd /home/vagrant/srcPkgs \
  && wget https://github.com/chipsalliance/verible/releases/download/v0.0-2776-gbaf0efe9/verible-v0.0-2776-gbaf0efe9-Ubuntu-22.04-jammy-x86_64.tar.gz \
  && mkdir -p verible \
  && tar xzvf verible-*-x86_64.tar.gz -C verible --strip-components 1)
else
  echo "verible already exists"
fi

# Install verilator from source - version v5.006
if [ ! -d "/home/vagrant/srcPkgs/verilator" ] 
then
  (mkdir -p /home/vagrant/srcPkgs \
  && cd /home/vagrant/srcPkgs \
  && git clone https://github.com/verilator/verilator)
else
  echo "verilator already exists"
fi
if [ "/usr/local/bin/verilator" ] 
then
  (mkdir -p /home/vagrant/srcPkgs \
  && cd /home/vagrant/srcPkgs \
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

# Env
if ! grep -q "Mase env" /home/vagrant/.bashrc; then
printf "\
\n# Mase env
\nexport LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LIBRARY_PATH \
\nexport PATH=/home/vagrant/.local/bin:/workspace/bin:/workspace/llvm/build/bin:\$PATH:/home/vagrant/srcPkgs/verible/bin \
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
\nautocmd BufWritePost *.py  silent! !yapf -i <afile> \
\nautocmd BufWritePost *.sv  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost *.v  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost * redraw! \
\n" >> /home/vagrant/.vimrc
fi

# Mase env
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export PATH=/home/vagrant/.local/bin:/workspace/bin:/workspace/llvm/build/bin:$PATH:/home/vagrant/srcPkgs/verible/bin
# Thread setup
export nproc=$(grep -c ^processor /proc/cpuinfo)
# Terminal color...
export PS1="[\\[$(tput setaf 3)\\]\t\\[$(tput setaf 2)\\] \u\\[$(tput sgr0)\\]@\\[$(tput setaf 2)\\]\h \\[$(tput setaf 7)\\]\w \\[$(tput sgr0)\\]] \\[$(tput setaf 6)\\]$ \\[$(tput sgr0)\\]"
export LS_COLORS='rs=0:di=01;96:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01'
