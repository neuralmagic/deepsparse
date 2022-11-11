FROM python:3.9

# install sparsezoo from github
ARG SPARSEZOO_BRANCH=main
RUN git clone https://github.com/neuralmagic/sparsezoo.git --depth 1 -b $SPARSEZOO_BRANCH
RUN pip3 install --no-cache-dir --upgrade -e "./sparsezoo"

# install deepsparse from github
ARG DEEPSPARSE_BRANCH=main
RUN git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b $DEEPSPARSE_BRANCH
RUN pip3 install --no-cache-dir --upgrade -e "./deepsparse[server]"

ARG CONFIG
COPY ${CONFIG} /data/config.yaml

ARG PORT
ENV PORT=$PORT
ENTRYPOINT deepsparse.server config /data/config.yaml --port $PORT
