# app.py
import streamlit as st
import io
import time
import os
import pandas as pd

from audio_generator import generate_audio
from recorder import start_webrtc, save_video
from response_logger import label_responses
from split_video_by_responses import split_video_by_responses
from extract_images_by_silence_times import extract_images_by_silence_times
from extract_pose_coordinates import extract_pose_coordinates

st.set_page_config(page_title="반응 실험", layout="centered")
st.title("🎧 집중 반응 실험")

# === 상태 초기화 ===
if "camera_ready" not in st.session_state:
    st.session_state.camera_ready = False
if "started" not in st.session_state:
    st.session_state.started = False
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "responses" not in st.session_state:
    st.session_state.responses = []
if "video_saved" not in st.session_state:
    st.session_state.video_saved = False
if "labels" not in st.session_state:
    st.session_state.labels = []

# === 1️⃣ 웹캠 띄워서 권한 먼저 받기 ===
st.subheader("1️⃣ 먼저 웹캠 권한을 허용해주세요")
ctx = start_webrtc()

if ctx and ctx.state.playing:
    st.session_state.camera_ready = True
    st.success("📷 웹캠 권한 허용됨!")

# === 2️⃣ 실험 시작 버튼 ===
if st.session_state.camera_ready and not st.session_state.started:
    st.subheader("2️⃣ 실험을 시작할 준비가 되셨나요?")
    if st.button("🚀 실험 시작"):
        audio, positions = generate_audio(duration_ms=60000)
        audio_bytes = io.BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)

        st.session_state.audio_bytes = audio_bytes
        st.session_state.silence_times = [round(p / 1000, 2) for p in positions]
        st.session_state.start_time = time.time()
        st.session_state.started = True
        st.rerun()

# === 3️⃣ 실험 진행 중 ===
if st.session_state.started:
    st.success("✅ 실험 중입니다. 무음이 들릴 때 아래 버튼을 눌러주세요.")
    st.audio(st.session_state.audio_bytes, format="audio/wav")

    elapsed = time.time() - st.session_state.start_time
    st.write(f"⏱️ 경과 시간: {round(elapsed, 2)}초")

    with st.form("reaction_form"):
        if st.form_submit_button("🎯 무음이라면 클릭!"):
            t = round(time.time() - st.session_state.start_time, 2)
            st.session_state.responses.append(t)
            st.success(f"📍 {t}초 반응 기록됨!")

    st.write("📌 반응 기록:", st.session_state.responses)

    # 4️⃣ 오디오 끝나면 자동 저장 및 분석
    if elapsed > 60 and not st.session_state.video_saved:
        st.info("🛑 실험 종료. 데이터 저장 및 분석 중...")
        os.makedirs("data", exist_ok=True)

        video_path = "data/full_experiment_video.avi"
        save_video(video_path)

        # 라벨링
        labels = label_responses(
            st.session_state.silence_times,
            st.session_state.responses
        )
        st.session_state.labels = labels

        df = pd.DataFrame({
            "silence_time": st.session_state.silence_times,
            "label": labels
        })
        df.to_csv("data/reaction_labels.csv", index=False)

        # 영상 클립 저장
        split_video_by_responses(
            video_path,
            st.session_state.silence_times,
            labels
        )

        # 이미지 추출 및 라벨 CSV 저장
        extract_images_by_silence_times(
            video_path=video_path,
            silence_times=st.session_state.silence_times,
            labels=labels,
            output_dir="data/images"
        )

        # 포즈 좌표 추출 및 병합 CSV 저장
        extract_pose_coordinates(
            image_dir="data/images",
            output_dir="data/poses"
        )

        st.session_state.video_saved = True
        st.success("✅ 실험 종료 및 데이터 저장 완료!")
        st.balloons()

        try:
            df_pose = pd.read_csv("data/poses/pose_labels.csv")
            st.markdown("### 🧍 이미지 + 포즈 라벨 미리보기")
            st.dataframe(df_pose.head())
        except:
            st.warning("포즈 라벨 CSV를 찾을 수 없습니다.")
