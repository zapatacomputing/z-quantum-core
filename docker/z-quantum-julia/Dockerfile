# Dockerfile for the default OpenPack docker image
FROM zapatacomputing/z-quantum-default:latest

ARG JULIA_MINOR_VERSION=1.5
ARG JULIA_PATCH_VERSION=1.5.3
WORKDIR /root

# get julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MINOR_VERSION}/julia-${JULIA_PATCH_VERSION}-linux-x86_64.tar.gz && \
    tar xf julia-${JULIA_PATCH_VERSION}-linux-x86_64.tar.gz && \
    ln -s /root/julia-${JULIA_PATCH_VERSION}/bin/julia /usr/local/bin/julia && \
    rm julia-${JULIA_PATCH_VERSION}-linux-x86_64.tar.gz

WORKDIR /app
ENTRYPOINT bash
