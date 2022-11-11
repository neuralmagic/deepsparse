import requests
import sys

from time import sleep


def run_inference(image, port):
    url = f'http://0.0.0.0:{port}/predict/from_files'
    path = [image]
    while True:
        sleep(2)
        files = [('request', open(img, 'rb')) for img in path]
        requests.post(url=url, files=files)


if __name__ == "__main__":
    run_inference(image=sys.argv[1], port=sys.argv[2])
