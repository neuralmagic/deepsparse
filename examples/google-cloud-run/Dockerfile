FROM python:3.9-slim

ARG config_path=./config.yaml

USER root

COPY ${config_path} /root/server-config.yaml

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY topo-four-core.json ./
ENV NM_ARCH_FILE=./topo-four-core.json

RUN python3 -m venv $VIRTUAL_ENV && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir "deepsparse-nightly[server]"

ENTRYPOINT deepsparse.server config /root/server-config.yaml --port 8080
