# Can be have the following values: all, server, transformers,
#  ultralytics, base
ARG VERSION
ARG DEPS="all"
ARG VENV="/venv"
ARG BRANCH

FROM python:3.8.16-slim-bullseye@sha256:322e38e3056cf87280ad80be615a6282aae768090f30d43d99abe413e1dd081a AS base
ARG VENV

# metainformation
LABEL org.opencontainers.image.version = "1.4"
LABEL org.opencontainers.image.authors = "Neuralmagic, Inc."
LABEL org.opencontainers.image.source = "https://github.com/neuralmagic/sparseml"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

RUN set -Eeuxo \
    && apt-get update \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git \
    && python3.8 -m venv $VENV \
    && $VENV/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

RUN \
    if [ -z "${VERSION}" ] ; then  \
    echo VERSION not provided && \
      if [ -z "${BRANCH}" ]; then  \
        git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b main; \
      else clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b $BRANCH; \
      fi; \
    else echo VERSION is $VERSION &&  \
      git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b release/${VERSION}; \
    fi;



FROM base AS container_branch_all
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG DEEPSPARSE_INSTALLABLE=./deepsparse[server,yolo,onnxruntime,yolov8]
RUN $VENV/bin/pip install --no-cache-dir "${DEEPSPARSE_INSTALLABLE}"

RUN deepsparse.transformers.run_inference --help \
    && deepsparse.image_classification.annotate --help


FROM base AS container_branch_server
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG DEEPSPARSE_INSTALLABLE=./deepsparse[server]
RUN $VENV/bin/pip install --no-cache-dir "${DEEPSPARSE_INSTALLABLE}"

FROM container_branch_server AS container_branch_transformers
RUN deepsparse.transformers.run_inference --help


FROM container_branch_server AS container_branch_torchvision
RUN deepsparse.image_classification.annotate --help


FROM base AS container_branch_ultralytics
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG DEEPSPARSE_INSTALLABLE=./deepsparse[server,yolo,yolov8]
RUN $VENV/bin/pip install --no-cache-dir "${DEEPSPARSE_INSTALLABLE}"


FROM base AS container_branch_base
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG DEEPSPARSE_INSTALLABLE=./deepsparse
RUN $VENV/bin/pip install --no-cache-dir "${DEEPSPARSE_INSTALLABLE}"

FROM base AS container_branch_dev
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG DEEPSPARSE_INSTALLABLE=./deepsparse[dev]
RUN $VENV/bin/pip install -e --no-cache-dir "${DEEPSPARSE_INSTALLABLE}"


FROM container_branch_${DEPS} AS build
RUN echo Build complete, going onto prod

FROM base as prod
ARG VENV
COPY --from=build $VENV $VENV
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import deepsparse'
ENTRYPOINT bash

