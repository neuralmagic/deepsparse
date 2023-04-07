# Can be have the following values: all, server, transformers,
#  ultralytics, base
ARG VERSION
ARG DEPS="all"
ARG VENV="/venv"
ARG BRANCH

FROM python:3.8.16-slim-bullseye@sha256:322e38e3056cf87280ad80be615a6282aae768090f30d43d99abe413e1dd081a AS base
ARG VENV

RUN set -Eeuxo \
    && apt-get update \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git \
    && python3.8 -m venv $VENV \
    && $VENV/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

RUN \
   if [ -z "${BRANCH}"] ; then  \
    echo Will install from pypi based on mode and version ; \
  else  \
    echo cloning from "${BRANCH}" &&  \
    git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b "${BRANCH}" ; \
  fi;



FROM base AS container_branch_base
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE=""
RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH && \
      $VENV/bin/pip install --no-cache-dir "./deepsparse[server]"; \
    elif [ "$MODE" = "nightly" ] ; then \
      echo Installing nightlies ... && \
      if [ -z "$VERSION" ] ; then \
        echo Installing latest nightlies ...  && \
        $VENV/bin/pip install --no-cache-dir "deepsparse-nightly[server]"; \
      else \
        echo Installing nightlies ... with $VERSION && \
        $VENV/bin/pip install --no-cache-dir "deepsparse-nightly[server]==$VERSION"; \
      fi; \
    elif [ -z "$VERSION" ] ; then \
      echo Installing latest deepsparse ... from pypi && \
      $VENV/bin/pip install --no-cache-dir "deepsparse[server]"; \
    else \
      echo Installing latest deepsparse version="$VERSION" from pypi && \
      $VENV/bin/pip install --no-cache-dir "deepsparse[server]==$VERSION"; \
    fi;




FROM base AS container_branch_all
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE=""
RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH && \
      $VENV/bin/pip install --no-cache-dir "./deepsparse[server,yolo,onnxruntime,yolov8]"; \
    elif [ "$MODE" = "nightly" ] ; then \
      echo Installing nightlies ... && \
      if [ -z "$VERSION" ] ; then \
        echo Installing latest nightlies ...  && \
        $VENV/bin/pip install --no-cache-dir "deepsparse-nightly[server,yolo,onnxruntime,yolov8]"; \
      else \
        echo Installing nightlies ... with $VERSION && \
        $VENV/bin/pip install --no-cache-dir "deepsparse-nightly[server,yolo,onnxruntime,yolov8]==$VERSION"; \
      fi; \
    elif [ -z "$VERSION" ] ; then \
      echo Installing latest deepsparse ... from pypi && \
      $VENV/bin/pip install --no-cache-dir "deepsparse[server,yolo,onnxruntime,yolov8]"; \
    else \
      echo Installing latest deepsparse version="$VERSION" from pypi && \
      $VENV/bin/pip install --no-cache-dir "deepsparse[server,yolo,onnxruntime,yolov8]==$VERSION"; \
    fi;

RUN deepsparse.transformers.run_inference --help \
    && deepsparse.image_classification.annotate --help


FROM base AS container_branch_dev
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE
RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH with editable mode && \
      $VENV/bin/pip install -e  "./deepsparse[dev]"; \
    else \
      echo Installing from main  with editable mode && \
      git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b main && \
      $VENV/bin/pip install -e "./deepsparse[dev]"; \
    fi;


FROM container_branch_${DEPS} AS build
RUN echo Build complete, going onto prod


FROM base as dev
ARG VENV
COPY --from=build $VENV $VENV
COPY --from=build deepsparse deepsparse
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import deepsparse'
ENTRYPOINT bash


FROM base as prod
ARG VENV
COPY --from=build $VENV $VENV
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import deepsparse'
ENTRYPOINT bash

