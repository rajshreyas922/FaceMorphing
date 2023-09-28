import argparse
import cv2
import feature_detection
from scipy.spatial import Delaunay
import face_morph
import matplotlib.pyplot as plt
import numpy as np

#python main.py --img0 images/bradpitt.png --img1 images/image013.png --alpha 0.5
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img0")
    parser.add_argument("--img1")
    parser.add_argument("--alpha")
    args = parser.parse_args()

    img0 = cv2.imread(args.img0)
    img1 = cv2.imread(args.img1)
    alpha = float(args.alpha)

    correspondences = feature_detection.generate_face_correspondences(img0, img1)
    _, img0, img1, points0, points1, list3 = correspondences
    list3 = np.array(list3)
    tri = Delaunay(list3)
    triangles = tri.simplices
    morphed_img = face_morph.morph(img0, img1, points0, points1, triangles, alpha=alpha)
    cv2.imwrite('morphed_image.jpg', morphed_img)

if __name__ == "__main__":
    main()
