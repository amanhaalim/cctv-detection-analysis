"""
Enhanced Feature Extraction Module
Improved appearance, behavioral, and spatial feature extraction
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import torch
from config_mvp import UNIFORM_DETECTION, REID_CONFIG


class FeatureExtractor:
    def __init__(self):
        self.uniform_detector = UniformDetector()
        if REID_CONFIG['enabled']:
            self.reid_model = self.load_reid_model()
        else:
            self.reid_model = None
    
    def load_reid_model(self):
        """Load ReID model for appearance features"""
        try:
            import torchreid
            model = torchreid.models.build_model(
                name=REID_CONFIG['model'],
                num_classes=1000,
                pretrained=True
            )
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            return model
        except ImportError:
            print("⚠️  torchreid not available, using color-based features only")
            return None
    
    def extract_appearance(self, crop: np.ndarray) -> Dict:
        """Extract comprehensive appearance features"""
        features = {}
        
        # Validate crop
        if crop is None or crop.size == 0:
            return features
        
        # Color histogram (always computed)
        features['color_histogram'] = self.compute_color_histogram(crop)
        
        # Uniform detection (if enabled)
        if UNIFORM_DETECTION['enabled']:
            uniform_result = self.uniform_detector.detect(crop)
            features['uniform_score'] = uniform_result['score']
            features['has_uniform'] = uniform_result['has_uniform']
            features['dominant_uniform_color'] = uniform_result.get('dominant_color', 'none')
        
        # ReID features (if available)
        if self.reid_model is not None:
            reid_features = self.extract_reid_features(crop)
            features['reid_embedding'] = reid_features
        
        # Texture features
        features['texture'] = self.compute_texture_features(crop)
        
        # Dominant color
        features['dominant_color'] = self.get_dominant_color(crop)
        
        return features
    
    def compute_color_histogram(self, crop: np.ndarray, bins: int = 16) -> np.ndarray:
        """
        Compute normalized color histogram in HSV space
        More robust than RGB for lighting variations
        """
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Focus on torso region (middle portion) for more consistent color
            h, w = crop.shape[:2]
            torso_region = hsv[h//3:2*h//3, :]
            
            # Compute histogram
            hist = cv2.calcHist([torso_region], [0, 1], None, [bins, bins],
                               [0, 180, 0, 256])
            
            # Normalize
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception as e:
            # Return default histogram on error
            return np.ones(bins * bins) / (bins * bins)
    
    def get_dominant_color(self, crop: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color using k-means clustering"""
        try:
            # Reshape image to be a list of pixels
            pixels = crop.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 3
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, 
                                            cv2.KMEANS_RANDOM_CENTERS)
            
            # Get most common cluster
            unique, counts = np.unique(labels, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]
            dominant_color = centers[dominant_idx].astype(int)
            
            return tuple(dominant_color)
        except:
            return (128, 128, 128)
    
    def extract_reid_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract ReID embedding for person re-identification"""
        try:
            # Resize to model input size
            resized = cv2.resize(crop, (128, 256))
            
            # Normalize
            img = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # Extract features
            with torch.no_grad():
                features = self.reid_model(img_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
        except Exception as e:
            return np.zeros(REID_CONFIG['feature_dim'])
    
    def compute_texture_features(self, crop: np.ndarray) -> Dict:
        """Compute texture features for uniform/clothing analysis"""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Focus on torso region
            h = gray.shape[0]
            torso = gray[h//3:2*h//3, :]
            
            # Compute features
            features = {}
            
            # Standard deviation (texture roughness)
            features['std'] = np.std(torso)
            
            # Edge density
            edges = cv2.Canny(torso, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Gradient magnitude
            gx = cv2.Sobel(torso, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(torso, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)
            features['gradient_mean'] = np.mean(magnitude)
            
            return features
        except:
            return {'std': 0, 'edge_density': 0, 'gradient_mean': 0}


class UniformDetector:
    """Enhanced uniform detection with better color matching"""
    
    def __init__(self):
        self.uniform_colors = UNIFORM_DETECTION.get('uniform_colors', [])
        self.threshold = UNIFORM_DETECTION.get('uniform_confidence_threshold', 0.5)
    
    def detect(self, crop: np.ndarray) -> Dict:
        """
        Detect uniform in person crop with improved algorithm
        Returns: {'score': float, 'has_uniform': bool, 'dominant_color': str}
        """
        if not self.uniform_colors or crop is None or crop.size == 0:
            return {'score': 0.0, 'has_uniform': False, 'dominant_color': 'none'}
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h, w = crop.shape[:2]
            
            # Focus on torso region (more reliable than full body)
            torso_hsv = hsv[h//3:2*h//3, :]
            
            max_score = 0.0
            dominant_color = 'none'
            
            for color_def in self.uniform_colors:
                # Create mask for uniform color
                mask = cv2.inRange(torso_hsv, color_def['lower'], color_def['upper'])
                
                # Calculate coverage
                uniform_pixels = np.sum(mask > 0)
                total_pixels = mask.size
                coverage = uniform_pixels / total_pixels if total_pixels > 0 else 0
                
                # Apply morphological operations to reduce noise
                kernel = np.ones((5,5), np.uint8)
                mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
                
                # Find largest connected component
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
                
                if num_labels > 1:
                    # Get largest component (excluding background)
                    largest_component_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                    largest_component_ratio = largest_component_area / total_pixels
                    
                    # Uniform should be a large, connected region
                    if largest_component_ratio > 0.15:
                        # Boost score if we have a large connected region
                        score = 0.4 * coverage + 0.6 * largest_component_ratio
                    else:
                        score = coverage * 0.8  # Penalize scattered matches
                else:
                    score = coverage * 0.7
                
                if score > max_score:
                    max_score = score
                    dominant_color = color_def['name']
            
            has_uniform = max_score > self.threshold
            
            return {
                'score': min(max_score, 1.0),
                'has_uniform': has_uniform,
                'dominant_color': dominant_color if has_uniform else 'none'
            }
        
        except Exception as e:
            return {'score': 0.0, 'has_uniform': False, 'dominant_color': 'none'}


class BehavioralAnalyzer:
    """Enhanced behavioral pattern analysis"""
    
    @staticmethod
    def compute_movement_features(positions: List[Tuple[int, int]], 
                                  velocities: List[float],
                                  directions: List[float]) -> Dict:
        """Compute comprehensive movement features"""
        features = {}
        
        if len(positions) < 2:
            return features
        
        # Speed analysis
        if velocities and len(velocities) > 0:
            features['avg_speed'] = np.mean(velocities)
            features['max_speed'] = np.max(velocities)
            features['min_speed'] = np.min(velocities)
            features['speed_variance'] = np.var(velocities)
            features['speed_std'] = np.std(velocities)
            
            # Speed percentiles
            features['speed_p25'] = np.percentile(velocities, 25)
            features['speed_p75'] = np.percentile(velocities, 75)
        
        # Direction changes (important for employee vs customer)
        if directions and len(directions) > 1:
            direction_changes = 0
            sharp_turns = 0
            
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                # Normalize to 0-π
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                
                if angle_diff > np.pi / 6:  # 30 degrees
                    direction_changes += 1
                if angle_diff > np.pi / 3:  # 60 degrees (sharp turn)
                    sharp_turns += 1
            
            features['direction_changes'] = direction_changes
            features['sharp_turns'] = sharp_turns
            features['direction_change_rate'] = direction_changes / len(directions)
            features['sharp_turn_rate'] = sharp_turns / len(directions)
        
        # Path analysis
        if len(positions) > 2:
            positions_arr = np.array(positions)
            
            # Displacement (straight-line distance)
            displacement = np.linalg.norm(positions_arr[-1] - positions_arr[0])
            features['displacement'] = displacement
            
            # Path length (total distance traveled)
            path_segments = np.diff(positions_arr, axis=0)
            path_length = np.sum(np.linalg.norm(path_segments, axis=1))
            features['path_length'] = path_length
            
            # Straightness index
            features['straightness'] = displacement / (path_length + 1e-6)
            
            # Tortuosity (inverse of straightness)
            features['tortuosity'] = path_length / (displacement + 1e-6)
            
            # Bounding box area (area covered by movement)
            min_x, min_y = np.min(positions_arr, axis=0)
            max_x, max_y = np.max(positions_arr, axis=0)
            bbox_area = (max_x - min_x) * (max_y - min_y)
            features['bbox_area'] = bbox_area
        
        # Stationary behavior (very important)
        if velocities and len(velocities) > 0:
            stationary_threshold = 0.8
            stationary_count = sum(1 for v in velocities if v < stationary_threshold)
            features['stationary_count'] = stationary_count
            features['stationary_ratio'] = stationary_count / len(velocities)
            
            # Moving vs stationary periods
            moving_count = len(velocities) - stationary_count
            features['moving_count'] = moving_count
            features['moving_ratio'] = moving_count / len(velocities)
        
        # Acceleration (change in velocity)
        if velocities and len(velocities) > 1:
            accelerations = np.diff(velocities)
            features['avg_acceleration'] = np.mean(np.abs(accelerations))
            features['max_acceleration'] = np.max(np.abs(accelerations))
        
        return features
    
    @staticmethod
    def compute_zone_features(zone_counts: Dict[str, int], 
                             total_frames: int,
                             zone_history: List[str] = None) -> Dict:
        """Compute zone-based behavioral features"""
        features = {}
        
        if total_frames == 0:
            return features
        
        # Zone occupancy ratios
        for zone_name, count in zone_counts.items():
            features[f'{zone_name}_ratio'] = count / total_frames
            features[f'{zone_name}_count'] = count
        
        # Number of unique zones visited
        features['unique_zones'] = len([z for z, c in zone_counts.items() if c > 0])
        
        # Dominant zone
        if zone_counts:
            dominant_zone = max(zone_counts.items(), key=lambda x: x[1])
            features['dominant_zone'] = dominant_zone[0]
            features['dominant_zone_ratio'] = dominant_zone[1] / total_frames
        
        # Zone transitions (if history available)
        if zone_history and len(zone_history) > 1:
            transitions = 0
            for i in range(1, len(zone_history)):
                if zone_history[i] != zone_history[i-1]:
                    transitions += 1
            
            features['zone_transitions'] = transitions
            features['zone_transition_rate'] = transitions / len(zone_history)
        
        return features
    
    @staticmethod
    def compute_temporal_features(frames: List[int], fps: float = 30.0) -> Dict:
        """Compute temporal features"""
        features = {}
        
        if len(frames) == 0:
            return features
        
        # Duration
        features['total_frames'] = len(frames)
        features['duration_sec'] = len(frames) / fps
        
        # Track continuity
        if len(frames) > 1:
            frame_array = np.array(frames)
            frame_diffs = np.diff(frame_array)
            
            features['avg_gap'] = np.mean(frame_diffs)
            features['max_gap'] = np.max(frame_diffs)
            features['min_gap'] = np.min(frame_diffs)
            
            # Number of gaps (frame_diff > 1)
            gaps = frame_diffs > 1
            features['num_gaps'] = np.sum(gaps)
            features['gap_ratio'] = np.sum(gaps) / len(frame_diffs)
            
            # Largest continuous segment
            continuous_segments = []
            current_segment = 1
            for diff in frame_diffs:
                if diff == 1:
                    current_segment += 1
                else:
                    continuous_segments.append(current_segment)
                    current_segment = 1
            continuous_segments.append(current_segment)
            
            features['max_continuous_frames'] = max(continuous_segments)
            features['avg_continuous_frames'] = np.mean(continuous_segments)
        
        return features


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    if a is None or b is None:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


def histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute histogram intersection similarity"""
    if hist1 is None or hist2 is None:
        return 0.0
    return np.sum(np.minimum(hist1, hist2))


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between histograms"""
    if hist1 is None or hist2 is None:
        return 1.0
    
    # Normalize
    hist1 = hist1 / (np.sum(hist1) + 1e-6)
    hist2 = hist2 / (np.sum(hist2) + 1e-6)
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1 * hist2))
    
    # Distance
    return -np.log(bc + 1e-6)