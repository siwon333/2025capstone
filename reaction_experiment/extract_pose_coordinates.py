# extract_pose_coordinates.py
import cv2
import os
import json
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_pose_coordinates(image_dir="data/images", output_dir="data/poses"):
    os.makedirs(output_dir, exist_ok=True)
    pose_data = []

    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)
        keypoints = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        pose_path = os.path.join(output_dir, filename.replace(".jpg", ".json"))
        with open(pose_path, "w") as f:
            json.dump(keypoints, f)

        pose_data.append({
            "filename": filename,
            "pose_file": pose_path
        })

    # 병합 CSV 생성
    if os.path.exists(os.path.join(image_dir, "image_labels.csv")):
        import pandas as pd
        df = pd.read_csv(os.path.join(image_dir, "image_labels.csv"))
        df_pose = pd.DataFrame(pose_data)
        merged = pd.merge(df, df_pose, on="filename", how="left")
        merged.to_csv(os.path.join(output_dir, "pose_labels.csv"), index=False)
