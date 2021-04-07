# Dockerfile for the default z-quantum-core docker image
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

# Set the default version of Python3 to Python 3.7 since z-quantum-core uses Python 3.7 features
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7

ENV PYTHONPATH="/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}"

# We fix pip version to 20.2.4 until the issues with dependency resolving in pip is solved.
RUN pip3 install pip==20.2.4
# Make sure to upgrade setuptools else z-quantum-core won't be installed because it uses find_namespace_packages
RUN pip3 install --upgrade setuptools
# Install Rigetti QVM
#RUN tar xjf sbcl-1.5.4-x86-64-linux-binary.tar.bz2
WORKDIR /root
RUN curl -O https://beta.quicklisp.org/quicklisp.lisp && \
    echo '(quicklisp-quickstart:install)'  | sbcl --load quicklisp.lisp
RUN git clone https://github.com/rigetti/quilc.git && \
                cd quilc && \
                git fetch && \
                git checkout v1.23.0 && \
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

RUN python3 -m pip install rpcq==3.6.0 \
                    pytest==5.3.5 \
                    networkx==2.4 \
                    pyquil==2.25.0 \
                    cirq==0.9.1 \
                    openfermion==1.0.0 \
                    openfermioncirq==0.4.0 \
                    qiskit==0.24 \
                    scipy==1.4.1 \
                    lea==3.2.0 \
                    numpy==1.18.1 \
                    pyyaml==5.1 \
                    overrides>=3.1.0 \
                    sympy==1.7 \
                    codecarbon==1.1.0 \
                    cvxpy==1.1.11


WORKDIR /app
ENTRYPOINT bash
