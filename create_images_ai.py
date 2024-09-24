import cv2
import os
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or use yolov8s.pt, yolov8m.pt, etc.

# Classes to detect (car, motorbike, bus, truck)
TARGET_CLASSES = [2, 3, 5, 7]  # COCO dataset IDs for car, motorbike, bus, and truck

def process_video(video_path, output_folder, frame_interval=15):
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_filename = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 15th frame
        if frame_count % frame_interval == 0:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Check if any of the target classes are detected
            if any(cls in TARGET_CLASSES for cls in results[0].boxes.cls.tolist()):
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                frame_filename = os.path.join(output_folder, f"{video_filename}_{timestamp}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")

        frame_count += 1

    cap.release()

def process_folder(videos_folder, output_folder, frame_interval=15):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, video_file)
        if os.path.isfile(video_path) and video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing {video_file}...")
            process_video(video_path, output_folder, frame_interval)

# Usage
videos_folder = 'videos'  # Replace with the path to your videos folder
output_folder = 'images'  # Replace with the path to save images

process_folder(videos_folder, output_folder)
