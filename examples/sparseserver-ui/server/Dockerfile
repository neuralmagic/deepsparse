# base docker image
FROM python:3.8 
# python:3.9-slim

WORKDIR /app

# install  deepsparse server
RUN pip install deepsparse[transformers,server]>=1.5.2

# copy server files. Could be replaced with a volume mapping 
COPY . /app/server

# command run at the entry point
CMD ["deepsparse.server", "--config-file", "server/config.yaml"]
