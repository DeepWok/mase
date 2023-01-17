# This is a Dockerfile to setup Jianyi's Centos 9 Docker environment
# Require:
# * A installed ssh key in the host machine
#   I know it is unsafe! That is why I only build from scratch
# * A path to Vitis HLS
#   Needed by most of my work

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

# Build the following packages from source:
RUN mkdir -p /dkrPkgs
# # cmake 3.20
# RUN cd /dkrPkgs \
#     && wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz \
#     && tar zxvf cmake-3.20.3.tar.gz \
#     && cd cmake-3.20.3 \
#     && ./bootstrap --prefix=/usr/local \
#     && make -j $(grep -c ^processor /proc/cpuinfo) \
#     && make install
# python 3.8
# RUN dnf install -y python3 python3-pip bzip2-devel libffi-devel xz-devel
RUN cd /dkrPkgs \
    && wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz \
    && tar xvf Python-3.8.3.tgz \
    && cd Python-3.8*/ \
    && ./configure --enable-optimizations --prefix=/usr/local \
    && make install

USER root

CMD ["bash"]

# Add dev-user
RUN getent group $GID || groupadd -g $GID dev-user
RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER dev-user

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --extra-index-url \
    https://download.pytorch.org/whl/cpu --user

WORKDIR workspace 
