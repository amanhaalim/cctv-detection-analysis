#!/usr/bin/env python3
"""
MVP Employee vs Customer Classifier
Fast, efficient, and production-ready
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
import os
import argparse
from datetime import datetime
from config_mvp import *; PROGRESS_BAR
from classifier_mvp import MVPClassifier, compute_movement_stats

PROGRESS_BAR = True

class RetailPersonTracker:
    """
    Optimized person tracking for retail environments
    Focus: Speed + Accuracy
    """
    
    def __init__(self, video_path: str, output_dir: str = "outputs"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        print("🔧 Loading YOLO model...")
        self.model = YOLO(DETECTION_MODEL)
        
        # Initialize classifier
        print("🔧 Initializing classifier...")
        self.classifier = MVPClassifier()
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate resize dimensions
        if RESIZE_WIDTH and RESIZE_WIDTH < self.width:
            self.resize_height = int(self.height * RESIZE_WIDTH / self.width)
            self.resize_width = RESIZE_WIDTH
            self.scale_factor = self.width / RESIZE_WIDTH
        else:
            self.resize_width = self.width
            self.resize_height = self.height
            self.scale_factor = 1.0
        
        # Tracking data
        self.tracks = {}
        
        # Scale zones to working resolution
        self.zone_polygons = {}
        if ZONES:
            for zone_name, zone_config in ZONES.items():
                polygon = zone_config['polygon']
                if self.scale_factor != 1.0:
                    scaled_polygon = polygon / self.scale_factor
                    self.zone_polygons[zone_name] = scaled_polygon.astype(np.int32)
                else:
                    self.zone_polygons[zone_name] = polygon.astype(np.int32)
        
        print(f"\n📹 Video Information:")
        print(f"   Resolution: {self.width}x{self.height} → {self.resize_width}x{self.resize_height}")
        print(f"   FPS: {self.fps:.1f} | Frames: {self.total_frames}")
        print(f"   Duration: {self.total_frames/self.fps:.1f}s")
        print(f"\n⚡ Processing Settings:")
        print(f"   Frame Skip: {FRAME_SKIP} (effective FPS: {self.fps/FRAME_SKIP:.1f})")
        print(f"   Zones: {len(ZONES)} configured")
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize and enhance frame"""
        # Resize
        if self.scale_factor != 1.0:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        
        # Enhance with CLAHE if enabled
        if CCTV_OPTIMIZATIONS['auto_enhance'] and CCTV_OPTIMIZATIONS['clahe_enabled']:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(
                clipLimit=CCTV_OPTIMIZATIONS['clahe_clip_limit'],
                tileGridSize=CCTV_OPTIMIZATIONS['clahe_tile_size']
            )
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def get_zone(self, point: tuple) -> str:
        """Determine which zone a point is in"""
        if not self.zone_polygons:
            return "unknown"
        
        x, y = int(point[0]), int(point[1])
        
        for zone_name, polygon in self.zone_polygons.items():
            result = cv2.pointPolygonTest(polygon, (x, y), False)
            if result >= 0:
                return zone_name
        
        return "unknown"
    
    def update_track(self, track_id: int, bbox: list, frame_idx: int, confidence: float):
        """Update or create track"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Initialize new track
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'id': track_id,
                'frames': [],
                'bboxes': [],
                'positions': [],
                'velocities': [],
                'zones': defaultdict(int),
                'confidences': []
            }
        
        track = self.tracks[track_id]
        
        # Update tracking info
        track['frames'].append(frame_idx)
        track['bboxes'].append([x1, y1, x2, y2])
        track['confidences'].append(confidence)
        
        # Calculate center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        track['positions'].append((cx, cy))
        
        # Calculate velocity
        if len(track['positions']) > 1:
            prev_pos = track['positions'][-2]
            dx = cx - prev_pos[0]
            dy = cy - prev_pos[1]
            velocity = np.sqrt(dx**2 + dy**2)
            track['velocities'].append(velocity)
        else:
            track['velocities'].append(0.0)
        
        # Update zone
        zone = self.get_zone((cx, cy))
        track['zones'][zone] += 1
    
    def process_video(self):
        """Main processing loop"""
        print("\n🎬 Processing video...")
        
        # Determine frames to process
        frames_to_process = range(0, self.total_frames, FRAME_SKIP)
        if MAX_FRAMES:
            frames_to_process = list(frames_to_process)[:MAX_FRAMES]
        
        # Setup video writer
        out = None
        if OUTPUT_CONFIG['save_annotated_video']:
            output_path = os.path.join(self.output_dir, 'annotated.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps / FRAME_SKIP,
                (self.resize_width, self.resize_height)
            )
        
        # Process frames
        if PROGRESS_BAR:
            pbar = tqdm(total=len(list(frames_to_process)), desc="Processing")
        
        for frame_idx in frames_to_process:
            # Read frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Preprocess
            frame = self.preprocess_frame(frame)
            
            # Run detection + tracking
            results = self.model.track(
                frame,
                persist=True,
                tracker=f"{TRACKER_TYPE}.yaml",
                classes=DETECTION_CLASSES,
                conf=DETECTION_CONF,
                iou=DETECTION_IOU,
                verbose=False
            )
            
            # Process detections
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for bbox, track_id, conf in zip(boxes, track_ids, confidences):
                    self.update_track(track_id, bbox, frame_idx, conf)
            
            # Visualize
            if OUTPUT_CONFIG['save_annotated_video']:
                vis_frame = self.visualize_frame(frame, frame_idx)
                out.write(vis_frame)
            
            if PROGRESS_BAR:
                pbar.update(1)
        
        if PROGRESS_BAR:
            pbar.close()
        
        self.cap.release()
        if out:
            out.release()
        
        print("✅ Video processing complete!")
    
    def visualize_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw visualizations"""
        vis = frame.copy()
        
        # Draw zones
        if OUTPUT_CONFIG['visualization']['show_zones']:
            for zone_name, polygon in self.zone_polygons.items():
                color = ZONES[zone_name]['color']
                
                # Semi-transparent fill
                overlay = vis.copy()
                cv2.fillPoly(overlay, [polygon], color)
                vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
                
                # Border
                cv2.polylines(vis, [polygon], True, color, 2)
                
                # Label
                M = cv2.moments(polygon)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(vis, zone_name, (cx-30, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # Draw tracks
        for track_id, track in self.tracks.items():
            if frame_idx not in track['frames']:
                continue
            
            idx = track['frames'].index(frame_idx)
            x1, y1, x2, y2 = track['bboxes'][idx]
            
            # Get current zone for color
            zone = self.get_zone(track['positions'][idx])
            
            if zone in ZONES:
                color = ZONES[zone]['color']
            else:
                color = (128, 128, 128)
            
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"ID:{track_id}"
            if OUTPUT_CONFIG['visualization']['show_track_ids']:
                cv2.putText(vis, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Trajectory
            if OUTPUT_CONFIG['visualization']['show_trajectories'] and len(track['positions']) > 1:
                traj_len = OUTPUT_CONFIG['visualization']['trajectory_length']
                start = max(0, idx - traj_len)
                
                for i in range(start, idx):
                    pt1 = track['positions'][i]
                    pt2 = track['positions'][i+1]
                    cv2.line(vis, pt1, pt2, color, 2)
        
        # Add frame info
        info_text = f"Frame: {frame_idx} | Tracks: {len([t for t in self.tracks.values() if frame_idx in t['frames']])}"
        cv2.putText(vis, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        return vis
    
    def finalize_and_classify(self):
        """Finalize tracks and classify"""
        print("\n🧠 Classifying tracks...")
        
        valid_tracks = {}
        
        for track_id, track in self.tracks.items():
            # Filter short tracks
            if len(track['frames']) < MIN_TRACK_LENGTH:
                continue
            
            # Filter low confidence
            if track['confidences']:
                avg_conf = np.mean(track['confidences'])
                if avg_conf < MIN_CONFIDENCE:
                    continue
            
            # Classify
            classification = self.classifier.classify_track(track)
            track['classification'] = classification
            
            # Compute movement stats
            track['movement_stats'] = compute_movement_stats(
                track['positions'],
                track['velocities']
            )
            
            valid_tracks[track_id] = track
        
        self.tracks = valid_tracks
        print(f"   ✓ Valid tracks: {len(valid_tracks)}")
    
    def save_results(self):
        """Save results to CSV"""
        print("\n💾 Saving results...")
        
        results = []
        for track_id, track in self.tracks.items():
            cls = track['classification']
            stats = track.get('movement_stats', {})
            
            result = {
                'track_id': track_id,
                'class': cls['class'],
                'confidence': cls['confidence'],
                'final_score': cls['final_score'],
                'zone_score': cls['scores']['zone'],
                'behavioral_score': cls['scores']['behavioral'],
                'temporal_score': cls['scores']['temporal'],
                'total_frames': len(track['frames']),
                'duration_sec': len(track['frames']) * FRAME_SKIP / self.fps,
                'avg_speed': stats.get('avg_speed', 0),
                'straightness': stats.get('straightness', 0),
            }
            
            # Add zone statistics
            for zone_name in ZONES.keys():
                result[f'zone_{zone_name}_count'] = track['zones'].get(zone_name, 0)
                total = len(track['frames'])
                result[f'zone_{zone_name}_ratio'] = track['zones'].get(zone_name, 0) / total if total > 0 else 0
            
            results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.output_dir, 'results.csv')
            df.to_csv(csv_path, index=False)
            print(f"   ✓ Results saved: {csv_path}")
            
            # Print summary
            self.print_summary(df)
            
            # Generate analytics dashboard
            self.generate_analytics_dashboard(csv_path)
    
    def generate_analytics_dashboard(self, csv_path: str):
        """Generate analytics dashboard"""
        try:
            from analytics_dashboard import RetailAnalytics, generate_html_dashboard
            
            print("\n📊 Generating analytics dashboard...")
            
            analytics = RetailAnalytics(csv_path, self.fps)
            
            # Save JSON report
            json_path = os.path.join(self.output_dir, 'analytics_report.json')
            analytics.save_report(json_path)
            print(f"   ✓ Analytics report: {json_path}")
            
            # Generate HTML dashboard
            html_path = os.path.join(self.output_dir, 'dashboard.html')
            generate_html_dashboard(json_path, html_path)
            print(f"   ✓ Dashboard: {html_path}")
            
        except Exception as e:
            print(f"   ⚠️  Dashboard generation failed: {e}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("📊 CLASSIFICATION SUMMARY")
        print("="*60)
        
        total = len(df)
        
        for class_name in ['Employee', 'Customer', 'Unknown']:
            count = len(df[df['class'] == class_name])
            pct = 100 * count / total if total > 0 else 0
            print(f"  {class_name:10s}: {count:3d} ({pct:5.1f}%)")
        
        print("\n" + "="*60)
        
        # Detailed stats
        if len(df) > 0:
            print(f"\nAverage Confidence: {df['confidence'].mean():.3f}")
            print(f"Average Duration:   {df['duration_sec'].mean():.1f}s")
        
        print(f"\n📁 Output Directory: {self.output_dir}/")
        print(f"   • results.csv - Classification results")
        if OUTPUT_CONFIG['save_annotated_video']:
            print(f"   • annotated.mp4 - Annotated video")
        print(f"   • analytics_report.json - Detailed analytics")
        print(f"   • dashboard.html - Interactive dashboard (open in browser)")


def main():
    parser = argparse.ArgumentParser(
        description="MVP Employee vs Customer Classifier for Retail"
    )
    parser.add_argument("--video", type=str, required=True,
                       help="Path to video file")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--no-video", action="store_true",
                       help="Skip saving annotated video (faster)")
    
    args = parser.parse_args()
    
    # Override config if requested
    if args.no_video:
        OUTPUT_CONFIG['save_annotated_video'] = False
    
    # Run pipeline
    print("\n" + "="*60)
    print("🏪 RETAIL PERSON CLASSIFIER - MVP")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")
    
    tracker = RetailPersonTracker(args.video, args.output)
    tracker.process_video()
    tracker.finalize_and_classify()
    tracker.save_results()
    
    print("\n✅ Processing complete!")
    print(f"⏱️  Results saved to: {args.output}/")


if __name__ == "__main__":
    main()