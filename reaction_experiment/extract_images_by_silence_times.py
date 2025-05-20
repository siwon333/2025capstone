import cv2
import os
import pandas as pd

def extract_images_by_silence_times(
    video_path: str,
    silence_times: list,
    labels: list,
    output_dir: str = "data/images",
    extract_every_n_frames: int = 5,
    pre_window: float = 2.0,
    post_window: float = 2.0,
):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_data = []

    for idx, silence_time in enumerate(silence_times):
        label = labels[idx]
        start_frame = int(max((silence_time - pre_window) * fps, 0))
        end_frame = int(min((silence_time + post_window) * fps, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame

        while frame_id < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_id - start_frame) % extract_every_n_frames == 0:
                filename = f"silence_{idx:03}_f{frame_id}.jpg"
                path = os.path.join(output_dir, filename)
                timestamp = round(frame_id / fps, 2)
                cv2.imwrite(path, frame)
                image_data.append({
                    "filename": filename,
                    "label": label,
                    "frame": frame_id,
                    "timestamp": timestamp,
                    "silence_idx": idx
                })
            frame_id += 1

    cap.release()

    # 저장
    df = pd.DataFrame(image_data)
    df.to_csv(os.path.join(output_dir, "image_labels.csv"), index=False)