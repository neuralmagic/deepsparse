# Setup the base image
FROM python:3.8-slim-bullseye

# Install git
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Activate venv
RUN python3.8 -m venv /venv
ENV PATH="venv/bin:$PATH"

RUN pip3 install --upgrade setuptools wheel

# Setup DeepSparse

ARG GIT_CHECKOUT
# if $GIT_CHECKOUT is not specified - just install from pypi
RUN if [ -z "${GIT_CHECKOUT}" ] ; then pip3 install --no-cache-dir --upgrade deepsparse[server] ; fi

# if $GIT_CHECKOUT is specified - clone, checkout $GIT_CHECKOUT, and install with -e
RUN if [ -n "${GIT_CHECKOUT}" ] ; then git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b $GIT_CHECKOUT; fi
RUN if [ -n "${GIT_CHECKOUT}" ] ; then pip3 install --no-cache-dir --upgrade -e "./deepsparse[server]" ; fi

