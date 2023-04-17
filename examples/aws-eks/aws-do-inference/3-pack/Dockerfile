ARG BASE_IMAGE

FROM $BASE_IMAGE

ARG MODEL_NAME
ARG MODEL_FILE_NAME
ARG PROCESSOR

LABEL description="Model $MODEL_NAME packed in a FastAPI Server container to run on $PROCESSOR"

COPY ./config.properties /app/config.properties

COPY ./3-pack/server-config.yaml /app/server/server-config.yaml

COPY ./3-pack/requirements.txt /app/server/requirements.txt

RUN pip install -r /app/server/requirements.txt

WORKDIR /app/server/

RUN echo "#! /bin/bash" > /app/server/serve
RUN echo "deepsparse.server config --port 8080 /app/server/server-config.yaml" >> /app/server/serve
RUN chmod 777 /app/server/serve

EXPOSE 8080

CMD ["bash", "/app/server/serve"]