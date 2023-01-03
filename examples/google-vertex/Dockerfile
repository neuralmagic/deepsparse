FROM python:3.8

WORKDIR /app

COPY server-config.yaml /app/

RUN pip install deepsparse[server]

CMD deepsparse.server --config-file server-config.yaml

EXPOSE 5543