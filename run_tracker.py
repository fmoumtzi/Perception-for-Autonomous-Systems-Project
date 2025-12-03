import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

from tracking.sort import Sort
from tracking.utils import (
    load_depth, 
    find_occlusion_rect, 
    detect_scene_change, 
    get_median_depth, 
    compute_iou, 
    is_overlapping
)

IMAGE_FOLDER = "output/seq2_rectified"
OCCLUSION_TEMPLATE = "tracking/occlusion.png"
VALID_CLASSES = [0, 1, 2, 3, 5, 7] 
CONF_THRESHOLD = 0.5
TRACKER_MAX_AGE = 30 
IOU_THRESHOLD = 0.15 
MIN_HITS = 1 

# Color Map for Classes (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # Person: Green
    1: (0, 255, 255),  # Bicycle: Yellow
    2: (255, 0, 0),    # Car: Blue
    3: (255, 165, 0),  # Motorcycle: Orange
    5: (0, 255, 0),    # Bus: Green
    7: (128, 0, 128)   # Truck: Purple
}
DEFAULT_COLOR = (255, 255, 255) # White for others

def main():
    """
    Main function to run object detection and tracking on a sequence of images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Detection_models/best.pt", help="YOLO model to use (default: yolov8s.pt)")
    parser.add_argument("--output_video", type=str, default="output/video/scene2.mp4", help="Path to save the output video (e.g., output.mp4)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    model = YOLO(args.model)
    # Initialize SORT tracker
    tracker = Sort(max_age=TRACKER_MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
    
    occ_img = cv2.imread(OCCLUSION_TEMPLATE)
    
    image_files = sorted(Path(IMAGE_FOLDER).glob("rect_left_*.png"))
    if not image_files:
        image_files = sorted(Path(IMAGE_FOLDER).glob("*.png"))

    last_occ_rect = None
    start_time = time.time()
    frame_count = 0
    prev_frame = None

    video_writer = None

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        
        # Initialize video writer if needed
        if args.output_video and video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output_video, fourcc, 30, (width, height))

        depth_map = load_depth(str(img_path))
        
        # Check for Scene Change
        if prev_frame is not None:
            # Threshold 0.99 is needed because the correlation only drops to ~0.985
            if detect_scene_change(prev_frame, frame, threshold=0.99):
                print(f"\n[SCENE CHANGE DETECTED] Resetting Tracker")
                tracker = Sort(max_age=TRACKER_MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
                last_occ_rect = None
        
        prev_frame = frame.copy()
    
        # Detect occlusion
        occ_rect = find_occlusion_rect(frame, occ_img, last_occ_rect)
        last_occ_rect = occ_rect
        
        if occ_rect:
            cv2.rectangle(frame, (occ_rect[0], occ_rect[1]), (occ_rect[2], occ_rect[3]), (50, 50, 50), 2)

        if True:
            # Run YOLO detection
            results = model(frame, classes=VALID_CLASSES, conf=CONF_THRESHOLD, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf, cls])
            detections = np.array(detections)
        else:
            detections = np.empty((0, 6))

        # Update tracker
        tracks = tracker.update(detections, occ_rect)

        # Visualize tracking results
        if True:
            for track in tracks:
                tx1, ty1, tx2, ty2, track_id, class_id = map(int, track)
                track_box = [tx1, ty1, tx2, ty2]
                
                color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
                
                matched_detection = False
                for det in detections:
                    if compute_iou(track_box, det[:4]) > 0.3:
                        matched_detection = True
                        break
                
                if matched_detection:
                    # Visible (Confirmed by YOLO)
                    d_val = get_median_depth(depth_map, track_box)
                    depth_str = f"{d_val:.2f}m" if d_val > 0 else "N/A"
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 2)
                    # Removed ID, only showing depth
                    cv2.putText(frame, f"{depth_str}", (tx1, ty1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # Unmatched (Prediction or Occluded)
                    pred_color = (0, 0, 255)
                    if occ_rect and is_overlapping(track_box, occ_rect, threshold=0.05):
                        # Occluded
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), pred_color, 2)
                        cv2.line(frame, (tx1, ty1), (tx2, ty2), pred_color, 1)
                        cv2.line(frame, (tx2, ty1), (tx1, ty2), pred_color, 1)
                        # Removed ID
                        cv2.putText(frame, f"(Pred)", (tx1, ty1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)

            if video_writer:
                video_writer.write(frame)

            cv2.imshow("Detection and Tracking", frame)
            if cv2.waitKey(1) == 27: # Hardcoded delay = 1ms 
                break
            
    if video_writer:
        video_writer.release()
        print(f"Video saved to {args.output_video}")

    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
