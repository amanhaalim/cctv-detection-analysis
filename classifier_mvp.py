"""
Streamlined MVP Classifier
Fast and efficient employee vs customer classification
"""
import numpy as np
from typing import Dict, List
from config_mvp import CLASSIFICATION, ZONES, BEHAVIORAL_FEATURES, TEMPORAL_CONFIG


class MVPClassifier:
    """
    Lightweight classifier using only zone, behavioral, and temporal signals
    No heavy dependencies, optimized for speed
    """
    
    def __init__(self):
        self.weights = CLASSIFICATION['weights']
        self.employee_threshold = CLASSIFICATION['employee_threshold']
        self.customer_threshold = CLASSIFICATION['customer_threshold']
    
    def classify_track(self, track: Dict) -> Dict:
        """
        Final classification for a complete track
        Returns: {'class': str, 'confidence': float, 'scores': dict}
        """
        total_frames = len(track['frames'])
        
        if total_frames == 0:
            return {
                'class': 'Unknown',
                'confidence': 0.5,
                'final_score': 0.5,
                'scores': {}
            }
        
        # Compute individual scores
        zone_score = self._compute_zone_score(track, total_frames)
        behavioral_score = self._compute_behavioral_score(track)
        temporal_score = self._compute_temporal_score(track, total_frames)
        
        # Weighted ensemble
        final_score = (
            self.weights['zone_based'] * zone_score +
            self.weights['behavioral'] * behavioral_score +
            self.weights['temporal'] * temporal_score
        )
        
        # Determine class
        if final_score >= self.employee_threshold:
            class_name = 'Employee'
            confidence = final_score
        elif final_score <= self.customer_threshold:
            class_name = 'Customer'
            confidence = 1.0 - final_score
        else:
            class_name = 'Unknown'
            confidence = 0.5
        
        return {
            'class': class_name,
            'confidence': confidence,
            'final_score': final_score,
            'scores': {
                'zone': zone_score,
                'behavioral': behavioral_score,
                'temporal': temporal_score
            }
        }
    
    def _compute_zone_score(self, track: Dict, total_frames: int) -> float:
        """
        Zone-based classification score
        Higher score = more likely employee
        """
        if not ZONES or total_frames == 0:
            return 0.5
        
        score = 0.0
        weight_sum = 0.0
        
        for zone_name, count in track['zones'].items():
            if zone_name not in ZONES or zone_name == 'unknown':
                continue
            
            ratio = count / total_frames
            zone_config = ZONES[zone_name]
            
            # Employee zones (high employee_weight) increase score
            # Customer zones (high customer_weight) decrease score
            zone_contribution = (
                zone_config['employee_weight'] - zone_config['customer_weight']
            )
            
            score += ratio * zone_contribution
            weight_sum += ratio
        
        # Normalize to 0-1 range
        # zone_contribution ranges from -0.8 to 0.9, so normalize accordingly
        score = (score + 0.8) / 1.7  # Maps [-0.8, 0.9] to [0, 1]
        
        return np.clip(score, 0, 1)
    
    def _compute_behavioral_score(self, track: Dict) -> float:
        """
        Behavioral pattern score based on movement
        Higher score = more likely employee
        """
        if len(track['positions']) < 5:
            return 0.5
        
        score = 0.5  # Start neutral
        
        # 1. Speed analysis
        if track['velocities']:
            avg_speed = np.mean(track['velocities'])
            
            if avg_speed < BEHAVIORAL_FEATURES['slow_speed_threshold']:
                score += 0.15  # Slow movement = employee
            elif avg_speed > BEHAVIORAL_FEATURES['fast_speed_threshold']:
                score -= 0.15  # Fast movement = customer
        
        # 2. Stationary behavior
        if track['velocities']:
            stationary_count = sum(
                1 for v in track['velocities'] 
                if v < BEHAVIORAL_FEATURES['stationary_threshold']
            )
            stationary_ratio = stationary_count / len(track['velocities'])
            
            if stationary_ratio > BEHAVIORAL_FEATURES['employee_stationary_ratio']:
                score += 0.15  # Often stationary = employee
            elif stationary_ratio < BEHAVIORAL_FEATURES['customer_stationary_ratio']:
                score -= 0.10  # Rarely stationary = customer
        
        # 3. Path straightness
        if len(track['positions']) > 3:
            positions = np.array(track['positions'])
            
            # Displacement (straight-line distance)
            displacement = np.linalg.norm(positions[-1] - positions[0])
            
            # Path length (total distance)
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            
            # Straightness index
            straightness = displacement / (path_length + 1e-6)
            
            if straightness > BEHAVIORAL_FEATURES['straightness_threshold']:
                score -= 0.10  # Straight path = customer (enter → browse → checkout → exit)
            else:
                score += 0.05  # Meandering path = employee (working around store)
        
        return np.clip(score, 0, 1)
    
    def _compute_temporal_score(self, track: Dict, total_frames: int) -> float:
        """
        Temporal score based on track duration
        Higher score = more likely employee (longer duration)
        """
        score = 0.5
        
        # Duration-based scoring
        if total_frames >= TEMPORAL_CONFIG['long_track_frames']:
            score = 0.80  # Very long = likely employee
        elif total_frames >= TEMPORAL_CONFIG['medium_track_frames']:
            score = 0.65  # Medium length = could be either, slight employee bias
        elif total_frames <= TEMPORAL_CONFIG['short_track_frames']:
            score = 0.30  # Very short = likely customer or noise
        else:
            # Linear interpolation between thresholds
            score = 0.5
        
        # Track continuity (fewer gaps = employee)
        if len(track['frames']) > 1:
            frame_array = np.array(track['frames'])
            gaps = np.diff(frame_array)
            num_large_gaps = np.sum(gaps > 5)  # Gaps larger than 5 frames
            
            if num_large_gaps > 3:
                score -= 0.10  # Many gaps = customer (in/out of frame)
            elif num_large_gaps == 0:
                score += 0.10  # No gaps = employee (continuous presence)
        
        return np.clip(score, 0, 1)


def compute_movement_stats(positions: List, velocities: List) -> Dict:
    """
    Compute basic movement statistics
    Useful for debugging and analysis
    """
    stats = {}
    
    if velocities:
        stats['avg_speed'] = np.mean(velocities)
        stats['max_speed'] = np.max(velocities)
        stats['min_speed'] = np.min(velocities)
    
    if len(positions) > 2:
        positions_arr = np.array(positions)
        displacement = np.linalg.norm(positions_arr[-1] - positions_arr[0])
        path_length = np.sum(np.linalg.norm(np.diff(positions_arr, axis=0), axis=1))
        
        stats['displacement'] = displacement
        stats['path_length'] = path_length
        stats['straightness'] = displacement / (path_length + 1e-6)
    
    return stats