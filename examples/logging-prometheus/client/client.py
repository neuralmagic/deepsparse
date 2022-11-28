import random, time, requests, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, default="http://0.0.0.0:5543/image_classification/predict/from_files")
parser.add_argument("--img1_path", type=str, default="client/goldfish.jpeg")
parser.add_argument("--img2_path", type=str, default="client/all_black.jpeg")

MAX_COUNT = 25
CUTOFF_INCREMENT = 0.1

def run_example(url, img1_path, img2_path):
    cutoff = 0.0

    count = 0
    while (cutoff <= 1.0):
        img_path = None
        if random.uniform(0, 1) < cutoff:
            img_path = img2_path
        else:
            img_path = img1_path
        
        files = [('request', open(img_path, 'rb'))]
        resp = requests.post(url=url, files=files)
        print(f"Sent File: {img_path}")
            
        if count == MAX_COUNT:
            count = 0
            cutoff += CUTOFF_INCREMENT
        count += 1
        time.sleep(0.25)

    count = 0
    while (cutoff >= 0.):
        img_path = None
        if random.uniform(0, 1) < cutoff:
            img_path = img2_path
        else:
            img_path = img1_path
        
        files = [('request', open(img_path, 'rb'))]
        resp = requests.post(url=url, files=files)
        print(f"Sent File: {img_path}")
    
        if count == MAX_COUNT:
            count = 0
            cutoff -= CUTOFF_INCREMENT
        count += 1
        time.sleep(0.25)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    run_example(args["url"], args["img1_path"], args["img2_path"])