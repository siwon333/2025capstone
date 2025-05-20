# video_splitter.py

import cv2
import os

def split_video_by_responses(
    video_path: str,
    response_times: list,
    pre_seconds: float = 1.0,
    post_seconds: float = 2.0,
    output_dir: str = "data/clips",
    labels: list = None
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for idx, t in enumerate(response_times):
        start_sec = max(t - pre_seconds, 0)
        end_sec = t + post_seconds

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        label_suffix = f"_{labels[idx]}" if labels else ""
        clip_path = os.path.join(output_dir, f"clip_{idx:03}{label_suffix}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(clip_path, fourcc, fps, (int(width), int(height)))

        for f in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"🎞 클립 저장: {clip_path}")

    cap.release()
    print("✅ 모든 클립 저장 완료.")
