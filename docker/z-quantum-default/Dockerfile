# Dockerfile for the default z-quantum-core docker image
FROM ubuntu:focal
WORKDIR /app
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install Python 3.8
RUN apt-get clean && apt-get update
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.8 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.8-dev

RUN apt-get -y install \
                wget \
                git \
                vim \
                htop \
                sbcl \
                curl \
                gfortran \
                clang-7 \
                libzmq3-dev \
                libz-dev \
                libblas-dev \
                liblapack-dev \
                openssh-client

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8

ENV PYTHONPATH="/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"

# Make sure to upgrade setuptools else z-quantum-core won't be installed because it uses find_namespace_packages
RUN python3 -m pip install --upgrade pip && python3 -m pip install --upgrade setuptools

# Build & install Rigetti QVM simulator
WORKDIR /root
RUN curl -O https://beta.quicklisp.org/quicklisp.lisp && \
    echo '(quicklisp-quickstart:install)'  | sbcl --load quicklisp.lisp
RUN git clone https://github.com/rigetti/quilc.git && \
                cd quilc && \
                git fetch && \
                git checkout v1.25.1 && \
                git submodule init && \
                git submodule update --init && \
                make && \
                mv quilc /usr/local/bin
RUN git clone https://github.com/rigetti/qvm.git && \
                cd qvm && \
                git fetch && \
                git checkout v1.17.1 && \
                make QVM_WORKSPACE=10240 qvm && \
                mv qvm /usr/local/bin

# Add SSH enhancments to allow GITHUB/SSH access
RUN true \
    && mkdir -p -m 0700 ~/.ssh \
    && ssh-keyscan github.com >> ~/.ssh/known_hosts \
    && chmod 600 $HOME/.ssh/known_hosts \
    && true

# Install z-quantum-core's dependencies, but not the library itself.
RUN true \
   && git clone https://github.com/zapatacomputing/z-quantum-core.git \
   && python3 -m pip install --no-cache-dir /root/z-quantum-core \
   && python3 -m pip uninstall -y z-quantum-core \
   && true

# Misc libraries that we'd like to have already preinstalled.
# codecarbon is used for tracking CO2 generation and is used inside the python3 runtime
# cvxpy is used in `z-quantum-qubo` and takes long time to install, so in order to reduce
# overhead, we decided to have it installed straight in the docker.
RUN python3 -m pip install \
        codecarbon==1.2.0 \
        cvxpy

WORKDIR /app
ENTRYPOINT bash
