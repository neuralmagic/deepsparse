FROM python:3.8

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
    pip3 install --no-cache-dir "deepsparse-nightly[server]"  # TODO: switch to deepsparse[server] >= 0.12

# create 'serve' command for sagemaker entrypoint
RUN mkdir /opt/server/
RUN echo "#! /bin/bash" > /opt/server/serve
RUN echo "deepsparse.server --port 8080 --config_file /root/server-config.yaml" >> /opt/server/serve
RUN chmod 777 /opt/server/serve

ENV PATH="/opt/server:${PATH}"
WORKDIR /opt/server

ENTRYPOINT ["bash", "/opt/server/serve"]
