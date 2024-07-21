import cv2, requests, json, argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Pass a name to a file to be annotated")
parser.add_argument('--image_path', default="test/images/a9f16c_2_9_png.rf.c048a60764e56735d7465cdec974d102.jpg")

MODEL_IMG_SIZE = 640
ENDPOINT_URL_FINETUNED = "http://localhost:5543/yolov5-s-finetuned/predict/from_files"
ENDPOINT_URL_COCO = "http://localhost:5543/yolov5-s-coco/predict/from_files"

def main(image_path):
    for endpoint_url in [ENDPOINT_URL_FINETUNED, ENDPOINT_URL_COCO]:
        im = cv2.imread(image_path)
        im_size = im.shape[:2]
        assert im_size[0] == im_size[1]
        scale_ratio = im_size[0] / MODEL_IMG_SIZE

        resp = requests.post(
            url=endpoint_url,
            files=[('request', open(image_path, 'rb'))]
        )

        boxes = json.loads(resp.text)['boxes'][0]
        for xmin, ymin, xmax, ymax in boxes:
            start_point = (int(xmin * scale_ratio), int(ymin * scale_ratio))
            end_point = (int(xmax * scale_ratio), int(ymax * scale_ratio))
            color = (0, 255, 0)
            thickness = 2
            im = cv2.rectangle(im, start_point, end_point, color, thickness)

        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.imshow(im)
        
        if endpoint_url == ENDPOINT_URL_FINETUNED:
            plt.savefig("annotated-finetuned.png")
        else:
            plt.savefig("annotated-coco.png")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.image_path)