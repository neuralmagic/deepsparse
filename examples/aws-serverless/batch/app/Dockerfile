# Pull the  python 3.8 base image as a runtime for Lambda
FROM public.ecr.aws/lambda/python:3.8

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3 -m venv $VIRTUAL_ENV && \
    pip install --no-cache-dir --upgrade pip && \
    python3.8 -m pip install -r requirements.txt

# Copy the earlier created trigger.py file to the container
COPY trigger.py ./

# Set the CMD for handler
CMD ["trigger.lambda_handler"]