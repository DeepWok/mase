# This Dockerfile configures a Docker environment that 
# contains all the required packages for the tool
FROM ubuntu:22.04
ARG UID
ARG GID
ARG VHLS_PATH
RUN echo "Group ID: $GID"
RUN echo "User ID: $UID"

USER root
RUN apt-get update -y && apt-get install apt-utils -y
# RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install basic packages 
RUN apt-get upgrade -y 
RUN apt-get update -y \
    && apt-get install -y clang cmake graphviz-dev libclang-dev \
                          pkg-config g++ libxtst6 xdg-utils \
                          libboost-all-dev llvm gcc ninja-build \
                          python3 python3-pip build-essential \
                          libssl-dev git vim wget htop sudo \
                          lld parallel clang-format clang-tidy \
                          libtinfo5 gcc-multilib libidn11-dev

# Install SystemVerilog formatter
RUN mkdir -p /srcPkgs \
    && cd /srcPkgs \
    && wget https://github.com/chipsalliance/verible/releases/download/v0.0-2776-gbaf0efe9/verible-v0.0-2776-gbaf0efe9-Ubuntu-22.04-jammy-x86_64.tar.gz \
    && mkdir -p verible \
    && tar xzvf verible-*-x86_64.tar.gz -C verible --strip-components 1
# Install verilator from source - version v5.006 
RUN apt-get update -y \
    && apt-get install -y git perl make autoconf flex bison \
                          ccache libgoogle-perftools-dev numactl \
                          perl-doc libfl2 libfl-dev zlib1g zlib1g-dev \
                          help2man
RUN mkdir -p /srcPkgs \
    && cd /srcPkgs \
    && git clone https://github.com/verilator/verilator \ 
    && unset VERILATOR_ROOT \
    && cd verilator \
    && git checkout v5.006 \
    && autoconf \
    && ./configure \
    && make -j \
    && make install

# Append any packages you need here
# RUN apt-get ...

CMD ["bash"]

# Add dev-user
RUN groupadd -o -g $GID dev-user
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

# Install PyTorch and Torch-MLIR
ENV PATH="${PATH}:/home/dev-user/.local/bin"
RUN pip3 install --user --upgrade pip 
RUN pip3 install -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html \
    --pre torchvision==0.16.0.dev20230512
RUN pip3 install onnx black toml GitPython colorlog cocotb[bus]==1.8.0 \
                    pytest pytorch-lightning transformers toml \
                    timm pytorch-nlp datasets ipython ipdb \
                    sentencepiece einops deepspeed pybind11 \
                    tabulate tensorboardx hyperopt
# Install Torch-MLIR and Pytorch
RUN mkdir -p /home/dev-user/srcPkgs \
    && cd /home/dev-user/srcPkgs \
    && wget https://github.com/llvm/torch-mlir/releases/download/snapshot-20230513.837/torch-2.1.0.dev20230512+cpu-cp310-cp310-linux_x86_64.whl \
    && wget https://github.com/llvm/torch-mlir/releases/download/snapshot-20230513.837/torch_mlir-20230513.837-cp310-cp310-linux_x86_64.whl \
    && pip3 install torch-*.whl torch_mlir-*.whl
# The standard build is not stable as the torch
# RUN pip3 install --pre torch-mlir torchvision \
#                  -f https://llvm.github.io/torch-mlir/package-index/ \
#                  --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Add environment variables
ENV vhls $VHLS_PATH
RUN printf "\
\nexport LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LIBRARY_PATH \
\n# Basic PATH setup \
\nexport PATH=/workspace/scripts:/workspace/hls/build/bin:/home/dev-user/.local/bin:/workspace/llvm/build/bin:\$PATH:/srcPkgs/verible/bin \
\n# Vitis HLS setup \
\nexport VHLS=${vhls} \
\n# source ${vhls}/Vitis_HLS/2020.2/settings64.sh \
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
\nalias grep='grep --color'\n" >> /home/dev-user/.bashrc
#Add vim environment
RUN printf "\
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
\n" >> /home/dev-user/.vimrc

# Entrypoint set up
WORKDIR workspace

