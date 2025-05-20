# image_extractor.py

import cv2
import os
import pandas as pd

def extract_images_by_silence_times(
    video_path: str,
    silence_times: list,     # 초 단위
    labels: list = None,     # 0/1 리스트 (optional)
    output_dir: str = "data/images",
    extract_every_n_frames: int = 5,
    pre_window: float = 2.0,
    post_window: float = 2.0
):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    label_data = []

    for idx, silence_time in enumerate(silence_times):
        label = labels[idx] if labels else None

        start_sec = max(silence_time - pre_window, 0)
        end_sec = silence_time + post_window

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        saved_count = 0

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (current_frame - start_frame) % extract_every_n_frames == 0:
                image_filename = f"silence_{idx:03}_f{current_frame}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                cv2.imwrite(image_path, frame)
                label_data.append((image_filename, label))
                saved_count += 1
            current_frame += 1

        print(f"📸 무음 {idx} → 이미지 {saved_count}장 저장 완료")

    cap.release()

    # ✅ 라벨 CSV 저장
    if label_data:
        df = pd.DataFrame(label_data, columns=["filename", "label"])
        csv_path = os.path.join(output_dir, "image_labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ 이미지 라벨 CSV 저장 완료: {csv_path}")
    else:
        print("⚠️ 이미지가 저장되지 않아 CSV 생성을 건너뜸.")
