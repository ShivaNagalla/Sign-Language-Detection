# extract_frames.py

import os
import cv2
import glob

def extract_frames(video_path, output_dir, roi=(70, 10, 195, 165)):
    x, y, w, h = roi
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[y:y+h, x:x+w]
        frame_filename = os.path.join(output_dir, f"{frame_num:04d}.png")
        cv2.imwrite(frame_filename, roi_frame)
        frame_num += 1
    
    cap.release()
    print(f"Extracted {frame_num} frames from {os.path.basename(video_path)}")

if __name__ == "__main__":
    
    input_video_dir = r"C:\Users\manug\Sign-Language-Detection\rwth-boston-104\videoBank\camera0"        # replace with your actual path
    output_root = r"C:\Users\manug\Sign-Language-Detection\data\frames"            # where to store frame folders
    os.makedirs(output_root, exist_ok=True)

    video_files = glob.glob(os.path.join(input_video_dir, "*.mpg"))
    print(f"Found {len(video_files)} video files in {input_video_dir}")
    for video_path in video_files:
        print(f"Processing {video_path}")
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_root, video_id)
        extract_frames(video_path, output_dir)

    for video_path in glob.glob(os.path.join(input_video_dir, "*.mpg")):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_root, video_id)
        extract_frames(video_path, output_dir)