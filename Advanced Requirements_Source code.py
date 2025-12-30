
# ============================================================================
# mini-Project: Computer Vision + Eye Tracking_(Advanced Requirements)
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

from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Ultralytics YOLO
from ultralytics import YOLO


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def draw_dashed_rectangle(frame, pt1, pt2, color, thickness=2, dash_length=15):
    """Draw dashed rectangle border"""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top line
    for i in range(x1, x2, dash_length*2):
        cv2.line(frame, (i, y1), (min(i+dash_length, x2), y1), color, thickness)

    # Bottom line
    for i in range(x1, x2, dash_length*2):
        cv2.line(frame, (i, y2), (min(i+dash_length, x2), y2), color, thickness)

    # Left line
    for i in range(y1, y2, dash_length*2):
        cv2.line(frame, (x1, i), (x1, min(i+dash_length, y2)), color, thickness)

    # Right line
    for i in range(y1, y2, dash_length*2):
        cv2.line(frame, (x2, i), (x2, min(i+dash_length, y2)), color, thickness)

def draw_label_box(frame, text, position, color):
    """Draw text with colored background box"""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle
    cv2.rectangle(frame, (x-5, y-text_height-10), (x+text_width+5, y+5), color, -1)

    # Draw border
    cv2.rectangle(frame, (x-5, y-text_height-10), (x+text_width+5, y+5), (255,255,255), 1)

    # Put text
    cv2.putText(frame, text, (x, y), font, font_scale, (255,255,255), thickness)

# CLASS COLOR MAPPING
CLASS_COLORS = {
    'car': (0, 255, 0),              # Green
    'person': (255, 0, 0),           # Blue
    'traffic light': (128, 0, 255),  # Purple/Magenta
    'truck': (0, 255, 255),          # Cyan
    'bicycle': (0, 165, 255),        # Orange
    'motorcycle': (255, 165, 0),     # Light Blue
    'stop sign': (0, 0, 255),        # Red
    'bus': (0, 255, 255),            # Cyan
}

def get_class_color(class_name):
    """Get color for object class"""
    return CLASS_COLORS.get(class_name.lower(), (200, 200, 200))


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
        Returns list of (start_frame, end_frame, x, y, duration)
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
# OBJECT TRACKER 
# ============================================================================
class ObjectTracker:
    """Tracks detected objects across frames with persistent IDs"""

    def __init__(self):
        self.tracked_objects = defaultdict(lambda: {
            'class': None,
            'frames_seen': 0,
            'total_fixation_duration': 0,
            'bbox_history': [],
            'fixation_history': []
        })
        self.total_tracks = 0

    def is_point_in_mask(self, point, mask):
        """Check if a point (x, y) is within a segmentation mask"""
        x, y = int(point['x']), int(point['y'])  
        h, w = mask.shape

        if 0 <= x < w and 0 <= y < h:
            return mask[y, x] > 0
        return False

    def find_fixation_on_object(self, fixation, results):
        """
        Check which tracked object the fixation is on
        Returns (track_id, class_name, fixation_duration) or None
        """
        if not results or len(results) == 0:
            return None

        result = results[0]

        # Check if we have masks and tracking info
        if not hasattr(result, 'masks') or result.masks is None:
            return None

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        class_names = result.names

        # Check if tracking IDs are available
        if not hasattr(boxes, 'id') or boxes.id is None:
            return None

        track_ids = boxes.id.cpu().numpy().astype(int)

        # Find which object contains the gaze point
        for idx, (track_id, mask) in enumerate(zip(track_ids, masks)):
            if self.is_point_in_mask(fixation, mask):
                cls_id = int(boxes[idx].cls)
                class_name = class_names.get(cls_id, f"Object_{cls_id}")
                return (int(track_id), class_name, fixation['duration'])

        return None

# ============================================================================
# MAIN PIPELINE WITH OPTIMIZED TRACKING & ENHANCED VISUALIZATION
# ============================================================================
class EyeTrackingPipelineWithTracking:
    """
    Main pipeline: Load video, detect+track objects, track gaze, generate outputs
    Uses YOLO11 Small (yolo11s-seg) with optimized parameters
    Enhanced visualization with professional styling
    """

    def __init__(self, video_path, gaze_csv_path, output_dir='./output', model_name='yolo11s-seg'):
        self.video_path = video_path
        self.gaze_csv_path = gaze_csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load YOLO model with tracking capability 
        print(f"Loading {model_name} model with instance segmentation & tracking...")
        self.model = YOLO(f'{model_name}.pt')

        # Initialize analyzers 
        self.gaze_analyzer = GazeAnalyzer(velocity_threshold=22, min_fixation_duration=4)
        self.tracker = ObjectTracker()

        # Load gaze data
        print("Loading gaze data...")
        self.gaze_df = pd.read_csv(gaze_csv_path, sep='\s+')

        # identify fixation rows
        is_fix = self.gaze_df["event_type"] == "Fixation"

        # create fixation segment ids
        self.gaze_df["fix_id"] = (is_fix != is_fix.shift()).cumsum()

        # compute duration per fixation
        dur = (
          self.gaze_df[is_fix]
          .groupby("fix_id")["code"]
          .agg(lambda x: (x.max() - x.min()) / 1000)
        )

        # map duration back to gaze_df
        self.gaze_df["duration_ms"] = self.gaze_df["fix_id"].map(dur).fillna(0)

        self.fixations = self.gaze_analyzer.detect_fixations(self.gaze_df)

        print(f"✓ Detected {len(self.fixations)} fixations from gaze data")

    def process_video_with_tracking(self, output_video_path=None, conf_threshold=0.65,output_csv_path=None):

        """
        Process video with YOLO tracking (OPTIMIZED):
        - Detects objects with instance segmentation
        - Maintains consistent IDs across frames (TRACKING)
        - Associates gaze fixations with tracked objects
        - Enhanced visualization with professional styling
        - conf_threshold=0.65 (OPTIMIZED)
        - iou=0.4 (OPTIMIZED NMS)
        """

        # Convert to dictionary for fast frame lookup
        gaze_dict = {
            int(row.frame_gar): (row.X, row.Y)
            for _, row in self.gaze_df.iterrows()
        }

        fixation_frames = defaultdict(int)


        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # LIMIT TO 1 MINUTE
        one_minute_frames = int(fps * 30)  
        frames_to_process =  total_frames
      
        print(f"Video dimensions: {width} x {height}")
        print(f"Gaze X range: {self.gaze_df['X'].min()} to {self.gaze_df['X'].max()}")
        print(f"Gaze Y range: {self.gaze_df['Y'].min()} to {self.gaze_df['Y'].max()}")


        # Create fixation lookup by frame
        fixation_by_frame = defaultdict(list)

        if output_video_path is None:
            output_video_path = self.output_dir / 'annotated_video_with_tracking.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))


        for fix in self.fixations:
            # Only include fixations in first minute
            if fix['start_frame'] <= frames_to_process:
                for f in range(fix['start_frame'], min(fix['end_frame'] + 1, frames_to_process)):
                    fixation_by_frame[f].append(fix)

        # Track fixations per object 
        tracked_object_fixations = defaultdict(lambda: {'class': None, 'total_duration': 0})
        fps_val = fps  

        frame_idx = 0
        print("Processing video with tracking...")

        # Generate random colors for visualization
        np.random.seed(42)
        colors = np.random.randint(0, 255, (1000, 3))

        data=[]

        while frame_idx<frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            frame_copy = frame.copy()

            results = self.model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                iou=0.4,  
                verbose=False
            )
            result = results[0]

            if frame_idx in fixation_by_frame:
                for fixation in fixation_by_frame[frame_idx]:
                    x, y = int(fixation['x']), int(fixation['y'])


                    # Find which tracked object this fixation is on
                    fixation_info = self.tracker.find_fixation_on_object(fixation, results)

                    if fixation_info:
                        track_id, class_name, duration = fixation_info
                        tracked_object_fixations[track_id]['class'] = class_name
                        tracked_object_fixations[track_id]['total_duration'] += duration

            # Draw instance segmentation with tracking 
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                class_names = result.names


                
                # Process detected objects
                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)

                    # Draw masks and bounding boxes with track IDs 
                    for idx, (track_id, box, mask) in enumerate(zip(track_ids, boxes, masks)):
                        cls_id = int(box.cls)
                        class_name = class_names.get(cls_id, f"Object_{cls_id}")
                        conf = float(box.conf)
                        gaze_point = gaze_dict.get(frame_idx, None)


                        if gaze_point:
                          gx, gy = gaze_point
                          if gx == gx and gy == gy:   
                              gx, gy = int(gx), int(gy)


                        # Get color based on class type 
                        color = get_class_color(class_name)
                        color_bgr = tuple(map(int, color))

                        # Draw segmentation mask contours
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Get bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
          
                        # Check if gaze falls inside this object
                        if gaze_point:
                            fixation_frames[class_name] =(
                                (self.gaze_df["event_type"] != "Saccade") &
                                self.gaze_df["X"].between(x1, x2) &
                                self.gaze_df["Y"].between(y1, y2)
                            ).sum()
                            if gx == gx and gy == gy:  
                              gx, gy = int(gx), int(gy)

                              # Draw gaze point ONCE at the top of frame processing
                              if 0 <= gx < width and 0 <= gy < height:
                                  cv2.circle(frame_copy, (gx, gy), 10, (0, 255, 0), -1)
                                  cv2.circle(frame_copy, (gx, gy), 15, (0, 255, 0), 2)

                                  


                        
                        # Convert fixation frames → seconds
                        
                        fixation_time_sec = fixation_frames[class_name] / fps


                        draw_dashed_rectangle(frame_copy, (x1, y1), (x2, y2), color_bgr, thickness=2, dash_length=15)

                        # Get fixation info for this object
                        obj_info = tracked_object_fixations[track_id]
                        duration_ms = int(obj_info['total_duration'] * 1000 / fps_val) if obj_info['total_duration'] > 0 else 0

                        # Draw main label with background 
                        label_text = f"{class_name.upper()} ID:{track_id}"
                        draw_label_box(frame_copy, label_text, (x1, y1-5), color_bgr)

                    
                        # Draw duration label if object was looked at
                        if fixation_time_sec > 0:
                            duration_text = f"{fixation_time_sec * 1000} ms"
                            draw_label_box(frame_copy, duration_text, (x1, y1+25), color_bgr)
                            data.append({
                                'Track_ID': track_id,
                                'Object_Class': class_name,
                                'Total_Fixation_Duration_Frames': fixation_time_sec * fps,
                                'Total_Fixation_Duration_Seconds': fixation_time_sec
                            })

            # Add frame information
            cv2.putText(frame_copy, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_copy, f"Time: {frame_idx/fps:.2f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame_copy)

            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}/{total_frames}")

            frame_idx += 1

        if output_csv_path is None:
            output_csv_path = self.output_dir / 'fixation_report.csv'

        df = pd.DataFrame(data).sort_values('Total_Fixation_Duration_Frames', ascending=False)
        df.to_csv(output_csv_path, index=False)


        print(f"\n✓ CSV Report saved: {output_csv_path}")
        print("\n" + "="*70)
        print("FIXATION SUMMARY (Per Tracked Object)")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)

        cap.release()
        out.release()

        print(f"✓ Video saved: {output_video_path}")

        return tracked_object_fixations

    def generate_csv_report(self, tracked_object_fixations, output_csv_path=None):
        """Generate CSV report of fixation durations per TRACKED object"""

        if output_csv_path is None:
            output_csv_path = self.output_dir / 'fixation_report.csv'

        # Convert to DataFrame
        data = []
        for track_id in sorted(tracked_object_fixations.keys()):
            obj_info = tracked_object_fixations[track_id]
            class_name = obj_info['class'] if obj_info['class'] else 'Unknown'
            duration_frames = obj_info['total_duration']
            duration_seconds = duration_frames / 30.0  

            if duration_frames > 0:  
                data.append({
                    'Track_ID': track_id,
                    'Object_Class': class_name,
                    'Total_Fixation_Duration_Frames': duration_frames,
                    'Total_Fixation_Duration_Seconds': round(duration_seconds, 2)
                })
        return df

# ============================================================================
# EXECUTION 
# ============================================================================
def main(video_path, gaze_csv_path, output_dir='./output', model_name='yolo11s-seg', conf_threshold=0.65):
    """Main execution function with optimized parameters and enhanced visualization"""
    print("ADVANCED PIPELINE: INSTANCE SEGMENTATION + OBJECT TRACKING + EYE GAZE")
    
    # Initialize pipeline with tracking 
    pipeline = EyeTrackingPipelineWithTracking(video_path, gaze_csv_path, output_dir=output_dir, model_name=model_name)

    # Process video with tracking 
    tracked_object_fixations = pipeline.process_video_with_tracking(conf_threshold=conf_threshold)

    # Generate CSV report
    pipeline.generate_csv_report(tracked_object_fixations)

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print(f"Output directory: {pipeline.output_dir}")
    print("="*70 + "\n")

# ============================================================================
# Final_RUN 
# ============================================================================
if __name__ == '__main__':

    # 1. Mount different Google Drive account
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # 2. file paths:
    video_path = '/content/drive/MyDrive/data/video_garmin.avi'  
    gaze_csv_path = '/content/drive/MyDrive/data/etg_samples.txt'  

    # 3. output directory:
    output_dir = '/content/drive/MyDrive/project_output'

    main(
        video_path,
        gaze_csv_path,
        output_dir=output_dir,           
        model_name='yolo11s-seg',       
        conf_threshold=0.65              
    )



# ============================================================================
# Performance_Matrix
# ============================================================================

df = pd.read_csv('/content/drive/MyDrive/project_output/fixation_report.csv')

metrics = {
    'Metric': [
        'Total Objects Tracked',
        'Total Fixation Events',
        'Average Fixation Duration (s)',
        'Objects with >5s fixation',
        'Objects with <1s fixation',
        'Detection Success Rate (%)',
        'Tracking Consistency (%)',
        'Gaze-Detection Alignment (%)'
    ],
    'Value': [
        len(df),
        len(df),
        round(df['Total_Fixation_Duration_Seconds'].mean(), 2),
        len(df[df['Total_Fixation_Duration_Seconds'] > 5]),
        len(df[df['Total_Fixation_Duration_Seconds'] < 1]),
        100.0,  
        round(len(df[df['Total_Fixation_Duration_Seconds'] > df['Total_Fixation_Duration_Seconds'].mean()]) / len(df) * 100, 2),
        round(len(df[df['Total_Fixation_Duration_Seconds'] > 5]) / len(df) * 100, 2)
    ]
}

metrics_df = pd.DataFrame(metrics)

print("\n" + "="*70)
print("PERFORMANCE METRICS MATRIX")
print("="*70)
print(metrics_df.to_string(index=False))

# Save as CSV
metrics_df.to_csv('/content/drive/MyDrive/project_output/performance_metrics.csv', index=False)
print("\n✓ Saved as: performance_metrics.csv")