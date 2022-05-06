# base docker image
FROM python:3.8

WORKDIR /app

# install streamlit library
RUN pip install streamlit==1.8.1

# copy client directory. Could be replaced with a volume mapping 
COPY . /app/client

# command run at the entry point
CMD ["streamlit", "run", "client/app.py"]
