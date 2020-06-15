# Dockerfile for the default OpenPack docker image
FROM zapatacomputing/z-quantum-default:latest

# get julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.0-linux-x86_64.tar.gz && \
    tar xf julia-1.3.0-linux-x86_64.tar.gz && \
    ln -s ~/julia-1.3.0/bin/julia /usr/local/bin/julia
    
WORKDIR /app
ENTRYPOINT bash
