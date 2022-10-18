# Use Python37
FROM python:3.9-slim

ARG config_path=./config.yaml

USER root

COPY ${config_path} /root/server-config.yaml

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /
RUN python3 -m venv $VIRTUAL_ENV && pip3 install --no-cache-dir --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT deepsparse.server config /root/server-config.yaml --port 8080
