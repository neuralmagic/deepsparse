# Pull the base image with python 3.8
FROM python:3.8

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    python3.8 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Create Model dir for model, and output dir for model output
RUN mkdir model
RUN mkdir output

# Download model
RUN sparsezoo.download zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --save-dir ./model

CMD ["python", "app.py"]