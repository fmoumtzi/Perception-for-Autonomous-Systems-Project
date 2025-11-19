import os
from pathlib import Path
import cv2
import numpy as np

print("CWD:", os.getcwd())

CALIB_DIR = "data/calib_data"
LEFT_PATTERN = "image_02/data/*.png"
RIGHT_PATTERN = "image_03/data/*.png"
OUT_FILE = "output/stereo_params.npz"

# List of possible board sizes to look for (cols, rows)
POSSIBLE_CHESSBOARD_SIZES = [(5, 7), (7, 11), (6, 10), (5, 15)]

SQUARE_SIZE: float = 0.0995


def preprocess_image(gray, alpha=1, beta=20, block_size=21, C=8):
    """
    Applies Histogram Equalization and Adaptive Thresholding.
    """
    gray_eq = cv2.equalizeHist(gray)
    gray_adjusted = cv2.convertScaleAbs(gray_eq, alpha=alpha, beta=beta)
    gray_thresh = cv2.adaptiveThreshold(gray_adjusted, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block_size, C)
    return gray_thresh


def get_object_points(board_size, square_size):
    """
    Generates the 3D object points (0,0,0), (1,0,0)...
    dynamically based on the specific board size detected.
    """
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def collect_image_pairs(calib_dir, left_pattern, right_pattern):
    calib_dir = Path(calib_dir)
    left_files = sorted(calib_dir.glob(left_pattern))
    right_files = sorted(calib_dir.glob(right_pattern))

    if len(left_files) == 0:
        print(f"[WARN] No images found in {calib_dir / left_pattern}")
        return []

    # Ensure we match valid pairs even if counts differ
    min_len = min(len(left_files), len(right_files))
    return list(zip(left_files[:min_len], right_files[:min_len]))


def detect_corners(img_gray):
    """
    Tries to find corners using multiple board sizes.
    1. Checks raw image.
    2. If failed, checks preprocessed image (Histogram Eq + Adaptive Thresh).
    """
    # Strategy 1: Try on raw grayscale image
    for size in POSSIBLE_CHESSBOARD_SIZES:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(img_gray, size, flags)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
            corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners_refined, size

    # Strategy 2: Try on preprocessed image
    img_processed = preprocess_image(img_gray)
    for size in POSSIBLE_CHESSBOARD_SIZES:
        # No flags needed here, preprocessing did the work
        ret, corners = cv2.findChessboardCorners(img_processed, size, None)

        if ret:
            # Refine on the original gray image for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
            corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners_refined, size

    return False, None, None


def main():
    os.makedirs(Path(OUT_FILE).parent, exist_ok=True)

    pairs = collect_image_pairs(CALIB_DIR, LEFT_PATTERN, RIGHT_PATTERN)
    print(f"Found {len(pairs)} image pairs. Starting detection...")

    objpoints = []  # 3D points in real world space
    imgpoints_left = []  # 2D points in left image plane
    imgpoints_right = []  # 2D points in right image plane

    image_size = None
    valid_pairs = 0

    for i, (lf, rf) in enumerate(pairs):
        imgL = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)

        if imgL is None or imgR is None:
            continue

        if image_size is None:
            image_size = (imgL.shape[1], imgL.shape[0])

        # Detect on Left and Right
        okL, cornersL, sizeL = detect_corners(imgL)
        okR, cornersR, sizeR = detect_corners(imgR)

        # Success only if both detected AND they found the SAME board size
        if okL and okR and (sizeL == sizeR):
            objp = get_object_points(sizeL, SQUARE_SIZE)

            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            valid_pairs += 1
            print(f"Pair {i}: Success (Board: {sizeL})")
        else:
            print(f"Pair {i}: Failed or Mismatch (L={sizeL}, R={sizeR})")

    if valid_pairs < 5:
        raise RuntimeError(f"Only {valid_pairs} valid pairs found. Need at least 5 for good calibration.")

    print("\n--- Calibrating Single Cameras ---")
    # Left Camera
    rmsL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
    print(f"Left Camera RMS Error:  {rmsL:.4f}")

    # Right Camera
    rmsR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)
    print(f"Right Camera RMS Error: {rmsR:.4f}")

    print("\n--- Calibrating Stereo System ---")
    # Fix intrinsics so they don't drift too much during stereo optimization
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    rms_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,K1, D1, K2, D2, image_size, criteria=criteria_stereo, flags=flags)

    print(f"Stereo RMS Error: {rms_stereo:.4f}")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (T):\n{T.T}")

    np.savez(
        OUT_FILE,
        image_size=image_size,
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        rms_stereo=rms_stereo)

    print(f"\nSaved calibration parameters to {OUT_FILE}")


if __name__ == "__main__":
    main()