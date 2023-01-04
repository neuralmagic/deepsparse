import requests

url = "http://localhost:5543/predict" # Server's port default to 5543

obj = {
    "question": "Who is Mark?",
    "context": "Mark is batman."
}
for i in range(1000):
    requests.post(url, json=obj)




