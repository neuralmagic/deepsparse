import random, time, requests, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, default="http://0.0.0.0:5543/image_classification/predict/from_files")
parser.add_argument("--img1_path", type=str, default="client/goldfish.jpeg")
parser.add_argument("--img2_path", type=str, default="client/all_black.jpeg")
parser.add_argument("--num_iters", type=int, default=25)
parser.add_argument("--prob_incr", type=float, default=0.1)

def send_random_img(url, img1_path, img2_path, prob_img2):
    img_path = ""
    if random.uniform(0, 1) < prob_img2:
        img_path = img2_path
    else:
        img_path = img1_path 
    
    files = [('request', open(img_path, 'rb'))]
    resp = requests.post(url=url, files=files)
    print(f"Sent File: {img_path}")
    
def main(url, img1_path, img2_path, num_iters, prob_incr):
    prob_img2 = 0.0
    iters = 0
    increasing = True
    
    while (increasing or prob_img2 > 0.0):
        send_random_img(url, img1_path, img2_path, prob_img2)
            
        if iters % num_iters == 0 and increasing:
            prob_img2 += prob_incr
        elif iters % num_iters == 0:
            prob_img2 -= prob_incr
        iters += 1

        if prob_img2 >= 1.0:
            increasing = False
            prob_img2 -= prob_incr

        time.sleep(0.25)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args["url"], args["img1_path"], args["img2_path"], args["num_iters"], args["prob_incr"])