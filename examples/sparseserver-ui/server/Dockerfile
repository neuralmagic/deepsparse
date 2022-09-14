# base docker image
FROM python:3.8

WORKDIR /app

# install  deepsparse server
RUN pip install deepsparse[server]>=0.11

# copy server files. Could be replaced with a volume mapping 
COPY . /app/server

# command run at the entry point
CMD ["deepsparse.server", "config_file", "server/config.yaml"]
