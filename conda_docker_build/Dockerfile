FROM debian:buster-slim AS build-base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -q && apt-get install -q -y libxml2 build-essential

FROM build-base AS cuda-102
RUN touch /root/cuda102
COPY cuda_10.2.89_440.33.01_linux.run /root
RUN /bin/bash /root/cuda_10.2.89_440.33.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-10.2 && \
	rm -f /root/cuda_10.2.89_440.33.01_linux.run

FROM build-base AS cuda-111
RUN touch /root/cuda111
COPY cuda_11.1.1_455.32.00_linux.run /root
RUN /bin/bash /root/cuda_11.1.1_455.32.00_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.1 && \
	rm -f /root/cuda_*_linux.run

FROM build-base AS cuda-113
RUN touch /root/cuda111
COPY cuda_11.3.1_465.19.01_linux.run /root
RUN /bin/bash /root/cuda_11.3.1_465.19.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.3 && \
	rm -f /root/cuda_*_linux.run


FROM build-base AS cuda-base
RUN touch /root/cuda
COPY --from=cuda-102 /usr/local/cuda-10.2 /usr/local/cuda-10.2
COPY --from=cuda-111 /usr/local/cuda-11.1 /usr/local/cuda-11.1
COPY --from=cuda-113 /usr/local/cuda-11.3 /usr/local/cuda-11.3

FROM debian:buster-slim AS mamba-base
ARG VERSION=0.11.3
RUN apt-get -q update && apt-get install -q -y \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/$VERSION | \
    tar -xj -C /tmp bin/micromamba


FROM cuda-base
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV MAMBA_ROOT_PREFIX=/opt/conda

RUN apt-get update -q && apt-get install git -yq

# Use bash in Dockerfile RUN commands and make sure bashrc is sourced when
# executing commands with /bin/bash -c
# Needed to have the micromamba activate command configured etc.
ENV BASH_ENV /root/.bashrc
SHELL ["/bin/bash", "-c"]

# Setting $BASH_ENV and the SHELL command will not result in .bashrc being sourced when
# you supply the program to run as an argument to the "docker run" command.
# Manually add directory for micromamba installed executables to PATH as a workaround.
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

COPY --from=mamba-base /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=mamba-base /tmp/bin/micromamba /bin/micromamba

RUN ln -s /bin/micromamba /bin/mamba && \
    ln -s /bin/micromamba /bin/conda && \
    ln -s /bin/micromamba /bin/miniconda && \
    /bin/micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX && \
    echo "micromamba activate base" >> /root/.bashrc

RUN mamba install -y -n base boa conda-build conda-verify -c defaults -c conda-forge
RUN mamba install -y -n base pytorch=1.8 -c pytorch
CMD ["/bin/bash"]
