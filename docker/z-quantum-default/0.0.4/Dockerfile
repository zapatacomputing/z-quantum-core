# Dockerfile for the default OpenPack docker image
FROM ubuntu
WORKDIR /app
USER root
RUN apt-get clean && apt-get update
# Install python, pip, and other utilities
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

# Set the default version of Python3 to Python 3.7 since OpenPack uses Python 3.7 features
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 && \
    update-alternatives --set python3 /usr/bin/python3.7

ENV PYTHONPATH="/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}"

# Make sure we have the latest version of pip
RUN pip3 install --upgrade pip
# Make sure to upgraade setuptools else OpenPack won't be installed because it uses find_namespace_packages
RUN pip3 install --upgrade setuptools
# Install Rigetti QVM
#RUN tar xjf sbcl-1.5.4-x86-64-linux-binary.tar.bz2
WORKDIR /root
RUN curl -O https://beta.quicklisp.org/quicklisp.lisp && \
    echo '(quicklisp-quickstart:install)'  | sbcl --load quicklisp.lisp
RUN git clone https://github.com/rigetti/quilc.git && \
                cd quilc && \
                git fetch && \
                git checkout v1.18.0 && \
                git submodule init && \
                git submodule update --init && \
                make && \
                mv quilc /usr/local/bin
RUN git clone https://github.com/rigetti/qvm.git && \
                cd qvm && \
                git fetch && \
                git checkout v1.17.0 && \
                make QVM_WORKSPACE=10240 qvm && \
                mv qvm /usr/local/bin

RUN python3 -m pip install rpcq==3.0.0 \
                    pytest==5.3.5 \
                    networkx==2.3 \
                    pyquil==2.17.0 \
                    cirq==0.7.0 \
                    openfermion==0.11.0 \
                    openfermioncirq==0.3.0 \
                    qiskit==0.18.3 \
                    cffi>=1.14.0 \
                    scipy==1.4.1 \
                    lea==3.2.0 \
                    numpy==1.18.1 \
                    werkzeug==0.14.1 \
                    flask==1.1.1 \
                    pyyaml==5.1 \
                    quantum-grove>=1.0.0

RUN python3 -m pip install qutip>=5.4.1

WORKDIR /app
ENTRYPOINT bash
