import requests, threading

NUM_THREADS = 2
URL = "http://localhost:8080/v2/models/text-classification-model/infer"
sentences = ["I hate using GPUs for inference", "I love using DeepSparse on CPUs"] * 100

def tfunc(text):
    inference_request = {
        "inputs": [
            {
                "name": "sequences",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text],
            },
        ]
    }   
    resp = requests.post(URL, json=inference_request).json()
    for output in resp["outputs"]:
        print(output["data"])


threads = [threading.Thread(target=tfunc, args=(sentence,)) for sentence in sentences[:NUM_THREADS]]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()