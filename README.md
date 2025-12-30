# mini-Project-Computer-Vision-Eye-Tracking_HCE-Lab
## Purpose: To demonstrate the ability to apply computer vision techniques (e.g., object detection, recognition, and tracking) to a human-centered engineering problem. 

## Objective: To develop a computer vision application that automatically annotates a driving video footage with the human driver’s eye-gaze behavior. Your application will (1) detect objects of interest (e.g., cars, pedestrians, traffic signs, and signals) and (2) calculate the duration the human driver looked at each object. To demonstrate the ability to apply computer vision techniques (e.g., object detection,  recognition, and tracking) to a human-centered engineering problem. 

![image alt](https://github.com/ono1025/mini-Project-Computer-Vision-Eye-Tracking_HCE-Lab/blob/main/Object_Detection_Eye_tracking.png)


| # | Type       | Name             | Description                                                                 |
|---|------------|------------------|-----------------------------------------------------------------------------|
| 1 | Video file | video_garmin.avi | Driving video captured by a camera attached to the car                     |
| 2 | Video file | video_etg.avi    | Driving video captured by eye-tracking glasses                              |
| 3 | Text file  | etg_samples.txt  | Gaze data containing the (x, y) coordinates of the driver’s gaze, aligned with frames in `video_etg.avi` |

## Core Solution
### Minimum Requirements

Model: YOLOv8 Nano (yolov8n.pt)/n
Detection: Cars only (COCO class 2)
Video: First 60 seconds of video_etg.avi
Output: Bounding boxes + fixation points
Key Insight: Simple velocity-based fixation detection filters meaningful gaze from rapid scanning

### Advanced Requirements

Model: YOLO11 Small with Instance Segmentation (yolo11s-seg.pt)
Detection: 8+ classes (cars, pedestrians, traffic lights, stop signs, etc.)
Video: Full duration of video_garmin.avi
Output: Instance segmentation masks + persistent object tracking IDs
Key Insight: Instance masks + persistent tracking enables accurate per-object fixation accumulation across frames
