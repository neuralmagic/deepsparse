# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY topo-four-core.json ./
ENV NM_ARCH_FILE=./topo-four-core.json

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3 -m venv $VIRTUAL_ENV && \
    pip install --no-cache-dir --upgrade pip && \
    python3.8 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Load the BERT model from SparseZoo and store it in the model directory
RUN mkdir model

# RUN sparsezoo.download zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/pruned80_quant-none-vnni --save-dir ./model
RUN sparsezoo.download zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --save-dir ./model

# Set the CMD to your handler
CMD ["app.lambda_handler"]