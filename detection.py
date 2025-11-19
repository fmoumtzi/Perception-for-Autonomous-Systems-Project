import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

RECT_IMG_DIR = "output/seq1_rectified"
DEPTH_DIR    = "output/seq1_rectified"   # where depth_XXXX.npy lives
OUT_DIR      = "output/seq1_bboxes"

IMAGE_PATTERN = "rect_left_*.png"
SCORE_THRESH = 0.5


def init_detector():
    # tiny model, fine for laptop and simple detection
    return YOLO("yolov8n.pt")


def detect_objects(img, model):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(rgb, verbose=False)[0]

    detections = []
    names = model.names  # class index -> label string

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))

        if score < SCORE_THRESH:
            continue

        # keep label if you want it later (for debugging / crops)
        detections.append([x1, y1, x2, y2, score, cls_id, label])

    return detections


def depth_for_bbox(depth, bbox):
    """
    depth: HxW float32 array (meters)
    bbox:  (x1, y1, x2, y2)
    returns: scalar depth (meters) or np.nan
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = int(round(x1)), int(round(y1))
    x2, y2 = int(round(x2)), int(round(y2))

    h, w = depth.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return np.nan

    patch = depth[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0)

    if not np.any(valid):
        return np.nan

    return float(np.median(patch[valid]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = init_detector()
    img_dir = Path(RECT_IMG_DIR)
    img_files = sorted(img_dir.glob(IMAGE_PATTERN))

    for img_path in img_files:
        stem = img_path.stem              # e.g. 'rect_left_0000'
        frame_id = stem.split("_")[-1]    # '0000'

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}, skipping.")
            continue

        # ---- load matching depth map ----
        depth_path = Path(DEPTH_DIR) / f"depth_{frame_id}.npy"
        if not depth_path.exists():
            print(f"Depth file {depth_path} not found, skipping frame {frame_id}.")
            continue

        depth = np.load(str(depth_path))  # H x W, meters

        # ---- detections ----
        dets = detect_objects(img, model)

        # save raw detections as numpy: each row [x1,y1,x2,y2,score,cls_id,depth]
        rows = []
        for d in dets:
            x1, y1, x2, y2, score, cls_id, label = d
            Z = depth_for_bbox(depth, (x1, y1, x2, y2))
            rows.append([x1, y1, x2, y2, score, cls_id, Z])

        if rows:
            arr = np.array(rows, dtype=np.float32)
        else:
            arr = np.zeros((0, 7), dtype=np.float32)

        np.save(Path(OUT_DIR) / f"bboxes_{frame_id}.npy", arr)

        # optional: save a visualization
        vis = img.copy()
        for x1, y1, x2, y2, score, cls_id, Z in arr:
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

            if np.isfinite(Z) and Z > 0:
                text = f"{Z:.1f}m"
            else:
                text = "?m"

            cv2.putText(
                vis,
                text,
                (x1i, max(0, y1i - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(str(Path(OUT_DIR) / f"bboxes_{frame_id}.png"), vis)

        print(f"Frame {frame_id}: saved {arr.shape[0]} boxes")

    print("Done.")


if __name__ == "__main__":
    main()
