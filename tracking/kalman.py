import numpy as np

def initialize_kalman():
    kalman = {
        "x": np.array([0, 0, 0, 0]),  # State vector [x, vx, y, vy]
        "P": 1000 * np.eye(4),  # Initial uncertainty
        "F": np.array([[1, 1, 0, 0],  # x_pos += x_vel
                       [0, 1, 0, 0],  # x_vel constant
                       [0, 0, 1, 1],  # y_pos += y_vel
                       [0, 0, 0, 1]]), # y_vel constant
        "u": np.zeros(4),  # External motion
        "H": np.array([[1, 0, 0, 0],  # Observe x position
                       [0, 0, 1, 0]]), # Observe y position
        "R": 10 * np.eye(2),  # Measurement uncertainty
        "I": np.eye(4)  # Identity matrix
    }
    return kalman

def update_kalman(kalman, Z):
    x, P, H, R, I = kalman["x"], kalman["P"], kalman["H"], kalman["R"], kalman["I"]
    
    # Measurement residual y
    y = Z - np.dot(H, x)
    
    # Residual covariance S
    S = np.dot(H, np.dot(P, H.T)) + R
    
    # Kalman gain K
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))

    # Update state estimate x
    x = x + np.dot(K, y)
    
    # Update uncertainty P
    P = np.dot(I - np.dot(K, H), P)
    
    kalman["x"], kalman["P"] = x, P
    return kalman

def predict_kalman(kalman, class_id=None):
    x, P, F, u = kalman["x"], kalman["P"], kalman["F"], kalman["u"]
    
    Q = np.eye(4) * 0.1  # small noise

    # Predict state x
    x = np.dot(F, x) + u

    # Hardcoded increase in velocity
    # Class-specific tuning:
    # 0: Person -> 1.02
    # 1: Bicycle (Cyclist) -> 1.05
    # 2, 3, 5, 7: Vehicles -> 1.12 (Faster acceleration)
    multiplier = 1.02
    if class_id is not None:
        if class_id == 1: # Bicycle
            multiplier = 1.05
        elif class_id in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
            multiplier = 1.12
        
    x[1] = x[1] * multiplier
    x[3] = x[3] * multiplier
    
    # Predict uncertainty P
    P = np.dot(F, np.dot(P, F.T)) + Q
    
    kalman["x"], kalman["P"] = x, P
    return kalman

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox, class_id=None):
        # Initialize custom dictionary-based Kalman Filter
        self.kf = initialize_kalman()
        
        # Set initial state from bbox
        # bbox is [x1, y1, x2, y2]
        # State x is [x_center, x_vel, y_center, y_vel]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w/2
        cy = bbox[1] + h/2
        
        self.kf["x"] = np.array([cx, 0, cy, 0])
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Store dimensions separately as the KF only tracks center x, y
        self.width = w
        self.height = h
        self.class_id = class_id

    def update(self, bbox):
        # This is the full update (hit + state update)
        self.hit()
        
        # Update dimensions
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w/2
        cy = bbox[1] + h/2
        
        self.width = w
        self.height = h
        
        # Update KF
        Z = np.array([cx, cy])
        self.kf = update_kalman(self.kf, Z)

    def hit(self):
        # Register a hit without updating the KF state (for occluded objects)
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        # Predict KF
        self.kf = predict_kalman(self.kf, self.class_id)
        
        w = self.width
        h = self.height
        cx = self.kf["x"][0]
        cy = self.kf["x"][2]
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Convert back to bbox [x1, y1, x2, y2]
        bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        self.history.append(np.array(bbox).reshape((1, 4)))
        return self.history[-1]

    def get_state(self):
        w = self.width
        h = self.height
        cx = self.kf["x"][0]
        cy = self.kf["x"][2]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2]).reshape((1, 4))
