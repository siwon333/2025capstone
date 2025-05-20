# split_video_by_responses.py

import cv2
import os

def split_video_by_responses(
    video_path: str,
    response_times: list,
    labels: list,
    pre_sec: float = 2.0,
    post_sec: float = 2.0,
    output_dir: str = "data/clips"
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for idx, center_sec in enumerate(response_times):
        label = labels[idx]
        start_frame = int(max((center_sec - pre_sec) * fps, 0))
        end_frame = int(min((center_sec + post_sec) * fps, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_path = os.path.join(output_dir, f"clip_{idx:03}_label{label}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    cap.release()
