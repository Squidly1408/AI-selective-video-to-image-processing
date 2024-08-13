import cv2
import os

def extract_frames_from_video(video_path, output_dir, frame_interval):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    count = 0
    extracted_count = 0
    success, frame = cap.read()
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while success:
        if count % frame_interval == 1:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        success, frame = cap.read()
        count += 1

    cap.release()

def process_videos(input_dir, output_dir, frame_interval=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_filename in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_filename)
        if os.path.isfile(video_path) and video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            extract_frames_from_video(video_path, output_dir, frame_interval)

if __name__ == "__main__":
    input_directory = "videos"
    output_directory = "images"
    frame_interval = 60  # Save one frame every 60 frames
    
    process_videos(input_directory, output_directory, frame_interval)
