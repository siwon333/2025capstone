import cv2
import av
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode

FRAME_QUEUE = queue.Queue()

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        FRAME_QUEUE.put(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def start_webrtc():
    return webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True
    )

def save_video(output_path="data/full_experiment_video.avi", max_frames=6000, fps=10.0, resolution=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    frame_count = 0

    while not FRAME_QUEUE.empty() and frame_count < max_frames:
        frame = FRAME_QUEUE.get()
        out.write(frame)
        frame_count += 1

    out.release()
    return frame_count
