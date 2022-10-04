FROM python:3.10-alpine
RUN pip install prometheus-client
RUN mkdir /app
ADD main.py /app/
EXPOSE 8000
CMD ["python", "/app/main.py"]
