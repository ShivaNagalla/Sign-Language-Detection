import cv2
import mediapipe as mp
import numpy as np
import os
import glob

mp_holistic = mp.solutions.holistic

def extract_keypoints_from_frames(frames_dir):
    keypoints_all = []
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))  # or jpg if your frames are jpg

    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        for frame_path in frames:
            image = cv2.imread(frame_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            keypoints = []

            # Pose landmarks
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0]*33*3)

            # Face landmarks
            if results.face_landmarks:
                for lm in results.face_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0]*468*3)

            # Left hand landmarks
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0]*21*3)

            # Right hand landmarks
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0]*21*3)

            keypoints_all.append(keypoints)

    return np.array(keypoints_all)

if __name__ == "__main__":
    frames_root = "data/frames"  # Folder containing video_id subfolders with frames
    output_dir = "data/keypoints"
    os.makedirs(output_dir, exist_ok=True)

    video_folders = sorted(os.listdir(frames_root))
    print(f"Found {len(video_folders)} videos (frame folders)")

    for video_id in video_folders:
        frames_dir = os.path.join(frames_root, video_id)
        print(f"Extracting keypoints from frames of {video_id}...")
        keypoints_seq = extract_keypoints_from_frames(frames_dir)
        np.save(os.path.join(output_dir, f"{video_id}_keypoints.npy"), keypoints_seq)
        print(f"Saved keypoints for {video_id} with shape {keypoints_seq.shape}")