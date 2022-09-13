FROM python:3.9

ARG GIT_BRANCH=main
RUN git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b $GIT_BRANCH
RUN pip3 install --no-cache-dir --upgrade -e "./deepsparse[server]"

ARG CONFIG=./sample-config.yaml
COPY $CONFIG /server-config.yaml
ENTRYPOINT ["deepsparse.server", "config", "--port", "8080", "/server-config.yaml"]
