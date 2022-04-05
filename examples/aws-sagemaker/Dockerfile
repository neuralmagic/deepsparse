FROM python:3.8-bullseye

ARG config_path=./config.yaml

USER root

RUN apt-get -qq -y update && \
    apt-get -qq -y upgrade && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*


COPY ${config_path} /root/server-config.yaml

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


RUN python3 -m venv $VIRTUAL_ENV && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir "deepsparse[server]"

ENTRYPOINT ["deepsparse.server", "--port", "8080", "--config_file", "/root/server-config.yaml"]