import requests
import sys

from time import sleep


def run_inference(text, port, ip):
    url = f'http://{ip}:{port}/predict'
    obj = {"sequences": text}

    while True:
        sleep(5)
        r = requests.post(url=url, json=obj)
        print(r.text)


if __name__ == "__main__":
    run_inference(text=sys.argv[1], port=sys.argv[2], ip=sys.argv[3])
