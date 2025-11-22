import cv2
import numpy as np
import os
import re

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def is_overlapping(box1, box2, threshold=0.1):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    if box1_area > 0:
        return (inter_area / box1_area) > threshold
    return False

def load_depth(image_path):
    base_dir = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    match = re.search(r"(\d+)", file_name)
    if match:
        frame_id = match.group(1)
        depth_path = os.path.join(base_dir, f"depth_{frame_id}.npy")
        if os.path.exists(depth_path):
            return np.load(depth_path)
    return None

def find_occlusion_rect(frame, template, last_rect=None):
    if template is None: return None
    
    # Optimization: If we found it before, search in a local neighborhood first
    if last_rect:
        x1, y1, x2, y2 = last_rect
        h, w = template.shape[:2]
        margin = 50
        
        # Define search ROI
        search_x1 = max(0, x1 - margin)
        search_y1 = max(0, y1 - margin)
        search_x2 = min(frame.shape[1], x2 + margin)
        search_y2 = min(frame.shape[0], y2 + margin)
        
        roi = frame[search_y1:search_y2, search_x1:search_x2]
        
        if roi.shape[0] > h and roi.shape[1] > w:
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > 0.4:
                top_left = (max_loc[0] + search_x1, max_loc[1] + search_y1)
                bottom_right = (top_left[0] + w, top_left[1] + h)
                return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    # Fallback: Full search with downscaling for speed
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    # Ensure template is not larger than small_frame
    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
         # If template is huge, resize it to a reasonable target size relative to original frame
         template = cv2.resize(template, (217, 225))
    
    small_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)

    if small_template.shape[0] > small_frame.shape[0] or small_template.shape[1] > small_frame.shape[1]:
        return None

    res = cv2.matchTemplate(small_frame, small_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val > 0.4: 
        # Scale back up
        top_left = (int(max_loc[0] / scale), int(max_loc[1] / scale))
        h, w = template.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    return None

def get_median_depth(depth_map, bbox):
    if depth_map is None: return -1.0
    x1, y1, x2, y2 = map(int, bbox)
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x1 >= x2 or y1 >= y2: return -1.0
    roi = depth_map[y1:y2, x1:x2]
    valid_pixels = roi[roi > 0] 
    if len(valid_pixels) == 0: return -1.0
    return np.median(valid_pixels)

def detect_scene_change(frame1, frame2, threshold=0.9999):
    if frame1 is None or frame2 is None:
        return False
    
    # Convert to HSV
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms
    # Using 50 bins for Hue, 60 for Saturation
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # If correlation is low, it's a scene change
    return correlation < threshold
