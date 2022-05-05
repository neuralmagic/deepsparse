# Setup the base image
FROM python:3.8-slim-bullseye

# Activate venv
RUN python3.8 -m venv /venv
ENV PATH="venv/bin:$PATH"

# Setup DeepSparse
RUN pip3 install deepsparse[server]


