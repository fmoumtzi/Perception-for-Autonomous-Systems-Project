import os
from pathlib import Path
import cv2
import numpy as np

CALIB_FILE = "config/calib_cam_to_cam.txt"

LEFT_CAM_ID = "02"
RIGHT_CAM_ID = "03"

LEFT_DIR = "data/raw_seq3/image_02/data"
RIGHT_DIR = "data/raw_seq3/image_03/data"
IMAGE_PATTERN = "*.png"

OUT_DIR = "output/seq3_rectified"

# Disparity settings
NUM_DISPARITIES = 128
BLOCK_SIZE = 9


def load_calib_cam_to_cam(path):
    """
    Parse calib_cam_to_cam.txt file into a dict of np.arrays.
    """
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue

            key, val = line.split(":", 1)
            vals_str = val.strip().split()

            try:
                vals = [float(x) for x in vals_str]
            except ValueError:
                continue

            data[key] = np.array(vals, dtype=np.float64)
    return data


def get_camera_params(data, cam_id):
    """
    Extract intrinsics/distortion/rectified params for a given camera ID.
    """
    suffix = f"_{cam_id}"
    K = data[f"K{suffix}"].reshape(3, 3)
    D = data.get(f"D{suffix}", None)
    if D is not None:
        D = D.reshape(-1, 1)

    R = data.get(f"R{suffix}", None)
    if R is not None:
        R = R.reshape(3, 3)
    T = data.get(f"T{suffix}", None)
    if T is not None:
        T = T.reshape(3, 1)

    R_rect = data.get(f"R_rect{suffix}", None)
    if R_rect is not None:
        R_rect = R_rect.reshape(3, 3)
    P_rect = data.get(f"P_rect{suffix}", None)
    if P_rect is not None:
        P_rect = P_rect.reshape(3, 4)

    S = data.get(f"S{suffix}", None)
    S_rect = data.get(f"S_rect{suffix}", None)
    size = None
    size_rect = None
    if S is not None:
        size = (int(S[0]), int(S[1]))  # (w, h)
    if S_rect is not None:
        size_rect = (int(S_rect[0]), int(S_rect[1]))

    return {
        "K": K,
        "D": D,
        "R": R,
        "T": T,
        "R_rect": R_rect,
        "P_rect": P_rect,
        "size": size,
        "size_rect": size_rect,
    }


def build_rectification_maps(cam_left, cam_right):
    """
    Build undistort+rectify maps for both cameras.
    """
    K1, D1, R_rect1, P_rect1, size_rect1 = (
        cam_left["K"],
        cam_left["D"],
        cam_left["R_rect"],
        cam_left["P_rect"],
        cam_left["size_rect"],
    )
    K2, D2, R_rect2, P_rect2, size_rect2 = (
        cam_right["K"],
        cam_right["D"],
        cam_right["R_rect"],
        cam_right["P_rect"],
        cam_right["size_rect"],
    )

    if size_rect1 is not None:
        image_size = size_rect1
    else:
        image_size = cam_left["size"]

    # Generate the rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R_rect1, P_rect1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R_rect2, P_rect2, image_size, cv2.CV_32FC1)

    return (map1x, map1y), (map2x, map2y), image_size


def compute_disparity(rect_left, rect_right, num_disp, block_size):
    gl = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # Ensure num_disp is divisible by 16
    num_disp = int(num_disp)
    num_disp = max(16, num_disp - (num_disp % 16))

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        disp12MaxDiff=1)

    disp_raw = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    disp_raw[disp_raw <= 0] = np.nan
    return disp_raw


def collect_raw_pairs(left_dir, right_dir, pattern):
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)

    left_files = sorted(left_dir.glob(pattern))
    right_files = sorted(right_dir.glob(pattern))

    if len(left_files) == 0 or len(left_files) != len(right_files):
        print(f"Found {len(left_files)} left and {len(right_files)} right images.")
        # Fallback to using the minimum common length
        min_len = min(len(left_files), len(right_files))
        return list(zip(left_files[:min_len], right_files[:min_len]))

    return list(zip(left_files, right_files))


def draw_epipolar_lines(img, num_lines=10):
    h, w = img.shape[:2]
    out = img.copy()
    step = max(1, h // num_lines)
    for y in range(step // 2, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
    return out


def get_fb_from_P(camL, camR):
    P_L = camL["P_rect"]
    P_R = camR["P_rect"]

    f = P_L[0, 0]
    # P_R[0,3] = -f * B
    B = -P_R[0, 3] / f
    return f, B


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    calib_data = load_calib_cam_to_cam(CALIB_FILE)

    camL = get_camera_params(calib_data, LEFT_CAM_ID)
    camR = get_camera_params(calib_data, RIGHT_CAM_ID)

    f, B = get_fb_from_P(camL, camR)
    (map1x, map1y), (map2x, map2y), image_size = build_rectification_maps(camL, camR)

    pairs = collect_raw_pairs(LEFT_DIR, RIGHT_DIR, IMAGE_PATTERN)
    print(f"Found {len(pairs)} raw stereo pairs. Processing...")

    for i, (lf, rf) in enumerate(pairs):
        imgL = cv2.imread(str(lf))
        imgR = cv2.imread(str(rf))

        if imgL is None or imgR is None:
            print(f"Could not read {lf} or {rf}, skipping.")
            continue

        # 1. Rectify full images (No Cropping)
        rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

        # 2. Check Rectification with Epipolar Lines
        rectL_vis = draw_epipolar_lines(rectL)
        rectR_vis = draw_epipolar_lines(rectR)

        disp = compute_disparity(rectL, rectR, NUM_DISPARITIES, BLOCK_SIZE)

        depth = f * B / disp

        # 4. Save Outputs
        outL = Path(OUT_DIR) / f"rect_left_{i:04d}.png"
        outR = Path(OUT_DIR) / f"rect_right_{i:04d}.png"
        outL_epi = Path(OUT_DIR) / f"epipolar_left{i:04d}.png"
        outR_epi = Path(OUT_DIR) / f"epipolar_right{i:04d}.png"
        outDepth  = Path(OUT_DIR) / f"depth_{i:04d}.npy"

        cv2.imwrite(str(outL), rectL)
        cv2.imwrite(str(outR), rectR)
        cv2.imwrite(str(outL_epi), rectL_vis)
        cv2.imwrite(str(outR_epi), rectR_vis)

        # save depth
        np.save(str(outDepth), depth)
        if i % 10 == 0:
            print(f"Processed {i}/{len(pairs)} pairs...")

    print("Done. output in:", OUT_DIR)


if __name__ == "__main__":
    main()