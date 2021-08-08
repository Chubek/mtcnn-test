from mtcnn import MTCNN
import cv2
import requests
import numpy as np
import sys

detector = MTCNN()

def detect_face(url):
    req = requests.get(url)
    print("Reqest made...")
    nparr = np.fromstring(req.content, np.uint8)
    print("Array created...")
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Image created...")
    detect = detector.detect_faces(img_np)
    print("Image detected...")
    print(detect)


if __name__ == "__main__":
    url = sys.argv[1]

    detect_face(url)