FROM python:3.9-slim

ARG config_path=./config.yaml

USER root

COPY ${config_path} /root/server-config.yaml

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY topo-four-core.json ./
ENV NM_ARCH_FILE=./topo-four-core.json

RUN python3 -m venv $VIRTUAL_ENV && \
    pip3 install --no-cache-dir --upgrade pip $$ \
    pip3 install --no-cache-dir "deepsparse-nightly[server]"

# ENTRYPOINT deepsparse.server config /root/server-config.yaml --port 8080

RUN mkdir /opt/server/
RUN echo "#! /bin/bash" > /opt/server/serve
RUN echo "deepsparse.server config --port 8080 /root/server-config.yaml" >> /opt/server/serve
RUN chmod 777 /opt/server/serve

ENV PATH="/opt/server:${PATH}"
WORKDIR /opt/server

ENTRYPOINT ["bash", "/opt/server/serve"]
