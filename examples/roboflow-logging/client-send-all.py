import requests, os, time

# the dataset we downloaded had 3 subsets in 3 folders
paths = []
for folder_name in ['test', 'valid', 'train']:
    path = f"{folder_name}/images/"
    paths += [path + img_name for img_name in os.listdir(path)]
    

# same URL for the endpoint as before
ENDPOINT_URL = "http://localhost:5543/yolov5-s-coco/predict/from_files"


# send each image to the endpoint
i = 0
print(f"Sending {len(paths)} images to the server")
for image_path in paths:
    if i % 60 == 0:
        print(i)
    i+=1
    
    resp = requests.post(
      url=ENDPOINT_URL,
      files=[('request', open(image_path, 'rb'))]
    )
    print(resp)
    time.sleep(1)