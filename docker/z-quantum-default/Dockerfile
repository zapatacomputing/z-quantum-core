# Dockerfile for the default z-quantum-core docker image
FROM ubuntu
WORKDIR /app
USER root
RUN apt-get clean && apt-get update

# Install Python 3.7
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.7 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.7-dev

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
                liblapack-dev


# ZQuantum requires Python 3.7. By installing `python3-pip` we have also installed 
# Python 3.8. We need to set back the Python version to 3.7.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7

ENV PYTHONPATH="/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}"

# Make sure to upgrade setuptools else z-quantum-core won't be installed because it uses find_namespace_packages
RUN python3 -m pip install --upgrade setuptools

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

# Install z-quantum-core's dependencies, but not the library itself.
RUN python3 -m pip install --no-cache git+git://github.com/zapatacomputing/z-quantum-core && \
    python3 -m pip uninstall -y z-quantum-core

# Misc libraries that we'd like to have already preinstalled.
RUN python3 -m pip install \
        codecarbon==1.2.0 \
        cvxpy

                    
WORKDIR /app
ENTRYPOINT bash
