import requests, json

ENDPOINT_URL = "http://localhost:5543/yolov5-s-coco/predict/from_files"
IMAGE_PATH = "test/images/4b770a_3_6_png.rf.f5d975605c1f73e1a95a1d8edc4ce5b1.jpg"

resp = requests.post(
  url=ENDPOINT_URL,
  files=[('request', open(IMAGE_PATH, 'rb'))]
)

print(json.loads(resp.text))