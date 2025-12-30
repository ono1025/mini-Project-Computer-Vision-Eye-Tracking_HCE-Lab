

# ============================================================================
# mini-Project: Computer Vision + Eye Tracking_(Minimum Requirements)
# ============================================================================
import subprocess
import sys

def install_packages():
    """Install all required packages"""
    packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'torchvision'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("✓ All packages installed successfully!")

install_packages()

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

print("✓ All imports successful!")

# ============================================================================
# GAZE ANALYSIS CLASS
# ============================================================================
class GazeAnalyzer:
    """Analyzes gaze data to detect fixations and saccades"""

    def __init__(self, velocity_threshold=22, min_fixation_duration=4):
        """
        velocity_threshold: pixels/frame - threshold to distinguish fixation from saccade
        min_fixation_duration: minimum frames to count as fixation
        """
        self.velocity_threshold = velocity_threshold
        self.min_fixation_duration = min_fixation_duration

    def detect_fixations(self, gaze_data):
        """
        Detects fixations from raw gaze points
        Returns list of fixation dictionaries
        """
        if len(gaze_data) < 2:
            return []

        # Extract coordinates
        coordinates = gaze_data[['X', 'Y']].values
        frames = gaze_data['frame_gar'].values

        # Calculate velocities (Euclidean distance between consecutive points)
        velocities = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))

        # Smooth velocities to reduce noise
        if len(velocities) > 5:
            velocities = savgol_filter(velocities, window_length=5, polyorder=2)

        fixations = []
        in_fixation = False
        fix_start = 0
        fix_coords = []

        for i, vel in enumerate(velocities):
            if vel < self.velocity_threshold:
                if not in_fixation:
                    in_fixation = True
                    fix_start = i
                    fix_coords = [coordinates[i]]
                else:
                    fix_coords.append(coordinates[i])
            else:
                if in_fixation:
                    duration = i - fix_start
                    if duration >= self.min_fixation_duration:
                        avg_coord = np.mean(fix_coords, axis=0)
                        fixations.append({
                            'start_frame': int(frames[fix_start]),
                            'end_frame': int(frames[i-1]),
                            'x': float(avg_coord[0]),
                            'y': float(avg_coord[1]),
                            'duration': duration
                        })
                    in_fixation = False
                    fix_coords = []

        # Handle fixation at end
        if in_fixation:
            duration = len(velocities) - fix_start
            if duration >= self.min_fixation_duration:
                avg_coord = np.mean(fix_coords, axis=0)
                fixations.append({
                    'start_frame': int(frames[fix_start]),
                    'end_frame': int(frames[-1]),
                    'x': float(avg_coord[0]),
                    'y': float(avg_coord[1]),
                    'duration': duration
                })

        return fixations

# ============================================================================
# OBJECT TRACKER FOR CARS ONLY
# ============================================================================
class CarTracker:
    """Tracks cars and calculates fixation duration"""

    def __init__(self):
        self.car_fixations = defaultdict(lambda: {'total_duration': 0, 'frames_seen': 0})

    def is_point_in_bbox(self, point, bbox):
        """Check if point is inside bounding box"""
        x, y = int(point['x']), int(point['y'])
        x1, y1, x2, y2 = map(int, bbox)

        return x1 <= x <= x2 and y1 <= y <= y2

    def find_car_at_fixation(self, fixation, boxes, box_ids):
        """Find if fixation point is on a car"""
        for idx, (box, car_id) in enumerate(zip(boxes, box_ids)):
            if self.is_point_in_bbox(fixation, box.xyxy[0]):
                return (int(car_id), fixation['duration'])

        return None

# ============================================================================
# PIPELINE
# ============================================================================
class MinimumRequirementsPipeline:
    """
    Pipeline:
    - Use first 1 minute of video_etg.avi
    - Detect CARS only
    - Use bounding boxes 
    """

    def __init__(self, video_path, gaze_csv_path, output_dir='./output_minimum'):
        self.video_path = video_path
        self.gaze_csv_path = gaze_csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load YOLO model 
        print("Loading YOLOv8n model for car detection...")
        self.model = YOLO('yolov8n.pt')  

        # Initialize analyzers
        self.gaze_analyzer = GazeAnalyzer(velocity_threshold=22, min_fixation_duration=4)
        self.car_tracker = CarTracker()

        # Load gaze data
        print("Loading gaze data...")
        self.gaze_df = pd.read_csv(gaze_csv_path, sep='\s+')
        self.fixations = self.gaze_analyzer.detect_fixations(self.gaze_df)

        print(f"✓ Detected {len(self.fixations)} fixations from gaze data")

    def process_first_minute(self, output_video_path=None, conf_threshold=0.65):
        """
        Process FIRST 1 MINUTE of video_etg.avi
        - Detect cars only
        - Use bounding boxes
        - Calculate fixation duration
        """

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # LIMIT TO 1 MINUTE
        one_minute_frames = int(fps * 60)  
        frames_to_process = min(one_minute_frames, total_frames)

        print(f"Video: {width}x{height} @ {fps} FPS")
        print(f"Processing FIRST 1 MINUTE: {frames_to_process} frames ({frames_to_process/fps:.1f}s)")

        if output_video_path is None:
            output_video_path = self.output_dir / 'annotated_video_1minute_cars_only.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        # Create fixation lookup by frame
        fixation_by_frame = defaultdict(list)
        for fix in self.fixations:
            # Only include fixations in first minute
            if fix['start_frame'] <= frames_to_process:
                for f in range(fix['start_frame'], min(fix['end_frame'] + 1, frames_to_process)):
                    fixation_by_frame[f].append(fix)

        # Track car fixations
        car_fixations = defaultdict(int)  

        frame_idx = 0
        print("Processing video...")

        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            frame_copy = frame.copy()

            # DETECT CARS ONLY
            results = self.model(frame, verbose=False, conf=conf_threshold)
            result = results[0]

            # Filter for cars only (class 2 in COCO = car)
            cars_found = False
            if result.boxes is not None:
                for idx, box in enumerate(result.boxes):
                    cls_id = int(box.cls)

                    # Only draw cars (COCO class 2 = car)
                    if cls_id == 2:
                        cars_found = True
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw bounding box for car (GREEN)
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label
                        label = f"CAR "
                        cv2.rectangle(frame_copy, (x1, y1-25), (x1+150, y1), (0, 255, 0), -1)
                        cv2.putText(frame_copy, label, (x1+5, y1-8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw gaze points
            if frame_idx in fixation_by_frame:
                for fixation in fixation_by_frame[frame_idx]:
                    x, y = int(fixation['x']), int(fixation['y'])

                    # Draw fixation point (GREEN circle)
                    cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(frame_copy, (x, y), 12, (0, 255, 0), 2)

                    # Check if looking at a car
                    if result.boxes is not None and cars_found:
                        for idx, box in enumerate(result.boxes):
                            cls_id = int(box.cls)
                            if cls_id == 2:  
                                if self.is_point_in_bbox(fixation, box.xyxy[0]):
                                    car_id = idx
                                    car_fixations[car_id] += fixation['duration']

                                    # Draw looking at car info
                                    cv2.putText(frame_copy, f"Looking at CAR",
                                              (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame_copy, "Looking at: (no car)",
                                  (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 165, 255), 2)

            # Add frame information
            cv2.putText(frame_copy, f"Frame: {frame_idx}/{frames_to_process}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_copy, f"Time: {frame_idx/fps:.2f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            out.write(frame_copy)

            if frame_idx % 25 == 0:
                print(f"  Frame {frame_idx}/{frames_to_process}")

            frame_idx += 1

        cap.release()
        out.release()

        print(f"✓ Video saved: {output_video_path}")

        return car_fixations

    def is_point_in_bbox(self, point, bbox):
        """Check if point is inside bounding box"""
        x, y = int(point['x']), int(point['y'])
        x1, y1, x2, y2 = map(int, bbox)

        return x1 <= x <= x2 and y1 <= y <= y2

# ============================================================================
# MAIN FUNCTION 
# ============================================================================
def main_minimum(video_path, gaze_csv_path, output_dir):
    """Main execution"""

    print("1-Minute video_etg.avi | Cars Only | Bounding Boxes")
    

    # Initialize pipeline
    pipeline = MinimumRequirementsPipeline(video_path, gaze_csv_path, output_dir=output_dir)

    # Process first minute
    pipeline.process_first_minute()


    print("\n" + "="*70)
    print("MINIMUM REQUIREMENTS PROCESSING COMPLETE!")
    print(f"Output directory: {pipeline.output_dir}")
    print("="*70 + "\n")

# ============================================================================
# Final Run
# ============================================================================
if __name__ == '__main__':
    
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # Set your file paths:
    video_path = '/content/video_etg.avi'  
    gaze_csv_path = '/content/etg_samples.txt'  

    # Set output directory
    output_dir = '/content/drive/My Drive/project_output'

    # Run 
    main_minimum(video_path, gaze_csv_path, output_dir)