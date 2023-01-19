# This Dockerfile configures a Centos 9 Docker environment that 
# contains all the required packages for the tool
FROM tgagor/centos:stream8
ARG UID
ARG GID

# Install basic packages
RUN dnf update -y
RUN dnf install -y https://extras.getpagespeed.com/release-latest.rpm \
    && dnf install -y gcc-c++ sudo vim openssh-clients \
                   wget curl-devel expat-devel gettext-devel zlib-devel \
                   perl-CPAN perl-devel git lld openssl openssl-devel \
		   glibc-devel bzip2-devel libffi-devel xz-devel \
		   htop parallel libjpeg-turbo-devel \
    && dnf --enablerepo=powertools install -y \
		   glibc-static libstdc++-static

# Install packages for LLVM
RUN dnf install -y llvm clang gcc-gfortran boost boost-devel \
                   svn llvm-devel graphviz \
    && dnf --enablerepo=powertools install -y \
		   ninja-build doxygen

# Build the following packages from source:
RUN mkdir -p /dkrPkgs
# cmake 3.20
RUN cd /dkrPkgs \
    && wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz \
    && tar zxvf cmake-3.20.3.tar.gz \
    && cd cmake-3.20.3 \
    && ./bootstrap --prefix=/usr/local \
    && make -j $(grep -c ^processor /proc/cpuinfo) \
    && make install
# python 3.10
RUN cd /dkrPkgs \
    && wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz \
    && tar xvf Python-3.10.2.tgz \
    && cd Python-3.10*/ \
    && ./configure --enable-optimizations --prefix=/usr/local \
    && make install

USER root

CMD ["bash"]

# Add dev-user
RUN getent group $GID || groupadd -g $GID dev-user
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

# Install PyTorch and Torch-MLIR
RUN pip3 install --user --upgrade pip \
    && pip3 install --pre torch-mlir torchvision -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    && pip3 install onnx

# Add environment variables
# PATH
RUN echo 'export PATH=/workspace/bin:/workspace/llvm/build/bin:$PATH' >>/home/dev-user/.bashrc
# Thread setup
RUN echo 'export nproc=$(grep -c ^processor /proc/cpuinfo)' >>/home/dev-user/.bashrc

WORKDIR workspace 
