# Setup the base image & install dependencies
FROM ubuntu:latest
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        git-all \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
        python3-pip \
        python3-dev \
        build-essential libssl-dev libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# Activate venv
RUN python3.8 -m venv /venv
ENV PATH="venv/bin:$PATH"

# Setup DeepSparse
RUN pip3 install deepsparse[server]

