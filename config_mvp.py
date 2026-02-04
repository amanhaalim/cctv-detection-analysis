"""
Optimized MVP Configuration for Retail Store Employee/Customer Classification
Fast, efficient, and accurate
"""
import numpy as np

# ==================== PERFORMANCE SETTINGS ====================
# Optimized for speed while maintaining accuracy
FRAME_SKIP = 2              # Process every 2nd frame (~15 FPS for 30 FPS video)
RESIZE_WIDTH = 960          # Lower resolution for faster processing
MAX_FRAMES = None           # Process full video (set to 300 for testing)

# ==================== DETECTION SETTINGS ====================
DETECTION_MODEL = "yolov8n.pt"   # Fastest YOLO model
DETECTION_CONF = 0.4        # Confidence threshold
DETECTION_IOU = 0.5         # NMS threshold
DETECTION_CLASSES = [0]     # Person class only

# ==================== TRACKING SETTINGS ====================
TRACKER_TYPE = "bytetrack"  # Fast and reliable
TRACK_BUFFER = 30           # Frame buffer for lost tracks
MIN_TRACK_LENGTH = 5        # Minimum frames to be valid track
MIN_CONFIDENCE = 0.4        # Minimum detection confidence
PROGRESS_BAR = True

# ==================== ZONE DEFINITIONS ====================
# Exact zones from your labeled store image
# Resolution appears to be approximately 1920x1080

ZONES = {
    "counter": {
        "polygon": np.array([
            [186, 355],
            [257, 326],
            [290, 413],
            [230, 451],
        ]),
        "color": (0, 255, 255),
        "employee_weight": 1.0,
        "customer_weight": 0.3
    },
    "behind_counter": {
        "polygon": np.array([
            [46, 403],
            [176, 357],
            [214, 461],
            [64, 516],
        ]),
        "color": (0, 165, 255),
        "employee_weight": 1.5,
        "customer_weight": 0.0
    },
    "entrance": {
        "polygon": np.array([
            [1527, 831],
            [1527, 402],
            [573, 827],
        ]),
        "color": (0, 255, 0),
        "employee_weight": 0.5,
        "customer_weight": 0.5
    },
    "customer_area": {
        "polygon": np.array([
            [260, 193],
            [654, 791],
            [1527, 390],
            [1518, 34],
            [255, 36],
        ]),
        "color": (255, 0, 0),
        "employee_weight": 0.2,
        "customer_weight": 1.0
    },
}

# Video resolution
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# ==================== CLASSIFICATION WEIGHTS ====================
# Simplified ensemble - only using what works
CLASSIFICATION = {
    "weights": {
        "zone_based": 0.70,      # Zones are primary signal
        "behavioral": 0.20,      # Movement patterns help
        "temporal": 0.10         # Duration matters
    },
    
    # Decision thresholds - optimized for retail
    "employee_threshold": 0.65,  # Score >= 0.65 → Employee
    "customer_threshold": 0.35,  # Score <= 0.35 → Customer
    # Between 0.35-0.65 → Unknown (when ambiguous)
}

# ==================== BEHAVIORAL FEATURES ====================
BEHAVIORAL_FEATURES = {
    # Speed (pixels per frame at original resolution)
    "slow_speed_threshold": 2.0,    # Employees move slower
    "fast_speed_threshold": 8.0,    # Customers move faster
    
    # Stationary behavior
    "stationary_threshold": 1.5,    # Velocity threshold
    "employee_stationary_ratio": 0.40,  # Employees stand still more
    "customer_stationary_ratio": 0.20,  # Customers keep moving
    
    # Path characteristics
    "straightness_threshold": 0.6,  # Customers have straighter paths
}

# ==================== TEMPORAL FEATURES ====================
TEMPORAL_CONFIG = {
    "short_track_frames": 30,   # < 30 frames → likely noise/customer
    "medium_track_frames": 90,  # 30-90 frames → could be either
    "long_track_frames": 180,   # > 180 frames → likely employee
}

# ==================== OUTPUT SETTINGS ====================
OUTPUT_CONFIG = {
    "save_annotated_video": True,
    "save_detailed_csv": True,
    "crop_samples": False,  # Disable to save space
    
    "visualization": {
        "show_zones": True,
        "show_trajectories": True,
        "trajectory_length": 20,
        "show_confidence": True,
        "show_track_ids": True,
        "font_scale": 0.5,
        "line_thickness": 2,
    }
}

# ==================== VIDEO ENHANCEMENT ====================
# Light preprocessing for better detection
CCTV_OPTIMIZATIONS = {
    "auto_enhance": True,
    "clahe_enabled": True,
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": (8, 8),
}

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # INFO, DEBUG, WARNING
PROGRESS_BAR = True