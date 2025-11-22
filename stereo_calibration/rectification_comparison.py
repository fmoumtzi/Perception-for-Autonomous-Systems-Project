import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

USER_PARAMS_FILE = "../output/stereo_params.npz"

RAW_IMG_PATH = "data/calib_data/image_02/data/0000000000.png"
GT_RECT_PATH = "../output/seq1_rectified/rect_left_0000.png"


def rectification(user_params_file, raw_img):
    d = np.load(user_params_file)
    K1, D1 = d["K1"], d["D1"]
    K2, D2 = d["K2"], d["D2"]
    R, T = d["R"], d["T"]

    h, w = raw_img.shape[:2]
    image_size = (w, h)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    # Generate Maps
    mapx, mapy = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1
    )

    # Apply Rectification
    rectified_img = cv2.remap(raw_img, mapx, mapy, cv2.INTER_LINEAR)

    return rectified_img, P1


def main():
    # 1. Load Images
    img_raw = cv2.imread(RAW_IMG_PATH)
    img_gt = cv2.imread(GT_RECT_PATH)

    if img_raw is None:
        print(f"Error: Could not find Raw Image at {RAW_IMG_PATH}")
        return
    if img_gt is None:
        print(f"Error: Could not find GT Image at {GT_RECT_PATH}")
        return

    img, P = rectification(USER_PARAMS_FILE, img_raw)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.title("Our Rectification")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 1 Row, 2 Columns, Plot #2 (Right)
    plt.subplot(1, 2, 2)
    plt.title("Computed with Ground Truth")
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()