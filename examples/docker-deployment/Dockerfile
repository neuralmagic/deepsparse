FROM python:3.9

# install sparsezoo from github
ARG SPARSEZOO_BRANCH=main
RUN git clone https://github.com/neuralmagic/sparsezoo.git --depth 1 -b $SPARSEZOO_BRANCH
RUN pip3 install --no-cache-dir --upgrade -e "./sparsezoo"

# install deepsparse from github
ARG DEEPSPARSE_BRANCH=main
RUN git clone https://github.com/neuralmagic/deepsparse.git --depth 1 -b $DEEPSPARSE_BRANCH
RUN pip3 install --no-cache-dir --upgrade -e "./deepsparse[server]"

# download the deployment directory
ARG MODEL_STUB
RUN sparsezoo.download --save-dir /model $MODEL_STUB

ARG TASK
ENV TASK=$TASK
ENTRYPOINT deepsparse.server task $TASK --model_path /model/deployment
