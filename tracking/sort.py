import numpy as np
from tracking.kalman import KalmanBoxTracker
from tracking.utils import iou_batch, linear_assignment

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Matches detections to existing trackers using IOU.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    # Compute IOU matrix
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Hungarian algorithm
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    """
    Simple Online and Realtime Tracking (SORT) implementation.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 6)), occlusion_rect=None):
        """
        Updates trackers with new detections and returns tracked objects.
        """
        self.frame_count += 1
        
        # Prediction
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # Predict next state
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Matching
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched
        for m in matched:
            det = dets[m[0], :5] # Take first 5 elements (x1,y1,x2,y2,conf)
            trk = self.trackers[m[1]]
            
            # Update class ID if available
            if dets.shape[1] > 5:
                trk.class_id = int(dets[m[0], 5])
            
            # Check for occlusion overlap
            is_occluded = False
            if occlusion_rect is not None:
                # Simple overlap check
                # det is [x1, y1, x2, y2, conf]
                # occlusion_rect is [x1, y1, x2, y2]
                dx1, dy1, dx2, dy2 = det[:4]
                ox1, oy1, ox2, oy2 = occlusion_rect
                
                xx1 = max(dx1, ox1)
                yy1 = max(dy1, oy1)
                xx2 = min(dx2, ox2)
                yy2 = min(dy2, oy2)
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h
                
                if inter_area > 0:
                    is_occluded = True
            
            if is_occluded:
               # If occluded, DO NOT update KF state, but keep alive
                trk.hit()
            else:
                # Standard update
                trk.update(det)

        # Create new tracks
        for i in unmatched_dets:
            det = dets[i, :]
            class_id = int(det[5]) if len(det) > 5 else None
            trk = KalmanBoxTracker(det[:4], class_id)
            self.trackers.append(trk)
            
        # Output gathering
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Keep alive if young enough AND seen at least once
            has_been_seen_enough = (trk.hits >= self.min_hits) or (self.frame_count <= self.min_hits)
            is_active = (trk.time_since_update <= self.max_age)
            
            if is_active and has_been_seen_enough:
                # Return [x1, y1, x2, y2, track_id, class_id]
                # Use -1 if class_id is None
                cls_id = trk.class_id if trk.class_id is not None else -1
                ret.append(np.concatenate((d, [trk.id + 1], [cls_id])).reshape(1, -1))

            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))
