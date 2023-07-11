# Pull the base image with python 3.10 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.10.2023.07.11.09

# Copy the requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Load the BERT model from SparseZoo and store it in the model directory
RUN mkdir model

RUN sparsezoo.download zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --save-dir ./model

# Set the CMD to your handler
CMD ["app.lambda_handler"]