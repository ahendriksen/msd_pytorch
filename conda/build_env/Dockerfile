FROM debian:buster-slim AS BUILDBASE
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -q && apt-get install -q -y libxml2 build-essential

FROM BUILDBASE AS CUDA110
RUN touch /root/cuda110
COPY cuda_11.0.3_450.51.06_linux.run /root
RUN /bin/bash /root/cuda_11.0.3_450.51.06_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.0 && \
	rm -f /root/cuda_11.0.3_450.51.06_linux.run


FROM BUILDBASE AS CUDA102
RUN touch /root/cuda102
COPY cuda_10.2.89_440.33.01_linux.run /root
RUN /bin/bash /root/cuda_10.2.89_440.33.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-10.2 && \
	rm -f /root/cuda_10.2.89_440.33.01_linux.run

FROM BUILDBASE AS CUDA101
RUN touch /root/cuda101
COPY cuda_10.1.243_418.87.00_linux.run  /root
RUN /bin/bash /root/cuda_10.1.243_418.87.00_linux.run --toolkit --silent --installpath=/usr/local/cuda-10.1 && \
	rm -f /root/cuda_10.1.243_418.87.00_linux.run

FROM BUILDBASE AS CUDA100
RUN touch /root/cuda100
COPY cuda_10.0.130_410.48_linux /root
RUN /bin/bash /root/cuda_10.0.130_410.48_linux --toolkit --silent --override && \
	rm -f /root/cuda_10.0.130_410.48_linux

FROM BUILDBASE AS CUDA92
RUN touch /root/cuda92
COPY cuda_9.2.148_396.37_linux /root
RUN /bin/bash /root/cuda_9.2.148_396.37_linux --toolkit --silent --override && \
	rm -f /root/cuda_9.2.148_396.37_linux

FROM BUILDBASE AS CUDABASE
RUN touch /root/cuda
COPY --from=CUDA110 /usr/local/cuda-11.0 /usr/local/cuda-11.0
COPY --from=CUDA102 /usr/local/cuda-10.2 /usr/local/cuda-10.2
COPY --from=CUDA101 /usr/local/cuda-10.1 /usr/local/cuda-10.1
COPY --from=CUDA100 /usr/local/cuda-10.0 /usr/local/cuda-10.0
COPY --from=CUDA92 /usr/local/cuda-9.2 /usr/local/cuda-9.2


FROM debian:buster-slim AS MAMBABASE
ARG VERSION=0.11.3
RUN apt-get -q update && apt-get install -q -y \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/$VERSION | \
    tar -xj -C /tmp bin/micromamba


FROM CUDABASE
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

COPY --from=MAMBABASE /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=MAMBABASE /tmp/bin/micromamba /bin/micromamba

RUN ln -s /bin/micromamba /bin/mamba && \
    ln -s /bin/micromamba /bin/conda && \
    ln -s /bin/micromamba /bin/miniconda && \
    /bin/micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX && \
    echo "micromamba activate base" >> /root/.bashrc

RUN mamba install -y -n base boa conda-build conda-verify -c defaults -c conda-forge
RUN mamba install -y -n base pytorch=1.8 -c pytorch
CMD ["/bin/bash"]
