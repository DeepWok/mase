#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script installs initial packages for both Docker containers 
# --------------------------------------------------------------------
set -o errexit
set -o pipefail
set -o nounset


apt-get update -y && apt-get install apt-utils -y
DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install basic packages
apt-get upgrade -y
apt-get update -y \
    && apt-get install -y clang graphviz-dev libclang-dev \
                          pkg-config g++ libxtst6 xdg-utils \
                          libboost-all-dev llvm gcc ninja-build \
                          python3 python3-pip build-essential \
                          libssl-dev git vim wget htop \
                          lld parallel clang-format clang-tidy \
                          libtinfo5 libidn11-dev unzip \
                          locales python3-sphinx graphviz

locale-gen en_US.UTF-8

# Install SystemVerilog formatter
mkdir -p /srcPkgs \
    && cd /srcPkgs \
    && wget https://github.com/chipsalliance/verible/releases/download/v0.0-2776-gbaf0efe9/verible-v0.0-2776-gbaf0efe9-Ubuntu-22.04-jammy-x86_64.tar.gz \
    && mkdir -p verible \
    && tar xzvf verible-*-x86_64.tar.gz -C verible --strip-components 1
# Install verilator from source - version v5.020
apt-get update -y \
    && apt-get install -y git perl make autoconf flex bison \
                          ccache libgoogle-perftools-dev numactl \
                          perl-doc libfl2 libfl-dev zlib1g zlib1g-dev \
                          help2man
# Install Verilator from source
mkdir -p /srcPkgs \
    && cd /srcPkgs \
    && git clone https://github.com/verilator/verilator \
    && unset VERILATOR_ROOT \
    && cd verilator \
    && git checkout v5.020 \
    && autoconf \
    && ./configure \
    && make -j 4 \
    && make install

# Install latest Cmake from source
mkdir -p /srcPkgs \
    && cd /srcPkgs \
    && wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc5/cmake-3.28.0-rc5.tar.gz \
    && mkdir -p cmake \
    && tar xzvf cmake-*.tar.gz -C cmake --strip-components 1 \
    && cd cmake \
    && ./bootstrap --prefix=/usr/local \
    && make -j 4 \
    && make install

# Append any packages you need here
# apt-get ...
apt-get update -y \
    && apt-get install -y clang-12

export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y python3.13 python3.13-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 300 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 100 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 200 \
    && update-alternatives --config python3

