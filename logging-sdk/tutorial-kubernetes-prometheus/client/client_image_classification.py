import requests
import sys

from time import sleep


def run_inference(image, port, ip):
    url = f'http://{ip}:{port}/predict/from_files'
    path = [image]

    while True:
        sleep(5)
        files = [('request', open(img, 'rb')) for img in path]
        r = requests.post(url=url, files=files)
        print(r.text)


if __name__ == "__main__":
    run_inference(image=sys.argv[1], port=sys.argv[2], ip=sys.argv[3])
