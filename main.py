import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from natsort import natsorted
import tempfile
import io
from moviepy.editor import ImageSequenceClip, AudioFileClip
import subprocess
import requests
import sys

st.set_page_config(layout="wide")
st.title("YYDS Dance Generator")


@st.cache_data()
def load_model():
    url = "https://ncku365-my.sharepoint.com/:u:/g/personal/nm6121030_ncku_edu_tw/Eede7zJZ2xpCroTGVxtyfDcB1QNLo9stAWBGJcTrdHKByw?e=4azRU3&Download=1"
    response = requests.get(url)

    # Ensure the request was successful
    response.raise_for_status()

    # Write the content of the request to a file
    with open("checkpoints/fish_pix2pix/latest_net_G.pth", "wb") as f:
        f.write(response.content)


# Ensure the directory exists
os.makedirs("checkpoints/fish_pix2pix", exist_ok=True)

if os.path.exists("checkpoints/fish_pix2pix/latest_net_G.pth"):
    print("Model already exists.")
else:
    st.spinner("Downloading model...")
    load_model()


# Uploader 放置在標題下方並置中
uploaded_file = st.file_uploader("選擇一個影片檔", type=["mp4", "avi", "mov"], label_visibility="visible")
progress_container = st.container()
if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.video(uploaded_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 初始化 MediaPipe Pose 和 Drawing utilities
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()
        mp_drawing_styles = mp.solutions.drawing_styles

        # 打開影片檔案
        cap = cv2.VideoCapture(tmp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 初始化空框和黑色背景圖像
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            black_background = np.zeros_like(frame_rgb)
        else:
            st.error("讀取影片檔的第一幀失敗。")
            cap.release()
            exit()

        frame_number = 0
        mp_holistic = mp.solutions.holistic
        mp_hands = mp.solutions.hands
        # 自定義手部 landmark 風格
        left_hand_landmark_style = mp_drawing.DrawingSpec(color=(0, 196, 235), thickness=2)
        right_hand_landmark_style = mp_drawing.DrawingSpec(color=(255, 142, 0), thickness=2)

        with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:

            progress_text = "Operation in progress. Please wait."
            progress_bar = progress_container.progress(0, text=progress_text)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                black_background = np.zeros_like(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)

                # 繪製手部和姿勢 landmarks
                mp_drawing.draw_landmarks(
                    black_background, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=right_hand_landmark_style
                )
                mp_drawing.draw_landmarks(
                    black_background, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=left_hand_landmark_style
                )
                mp_drawing.draw_landmarks(
                    black_background,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                image_path = os.path.join(tmp_dir, f"frame_{frame_number}.png")
                print(image_path)
                cv2.imwrite(image_path, black_background)

                frame_number += 1
                progress = frame_number / total_frames
                progress_bar.progress(progress, text=progress_text)

        cap.release()

        if frame_number > 0:
            image_paths = natsorted([os.path.join(tmp_dir, f"frame_{i}.png") for i in range(frame_number)])

            # Create video clip from image sequence
            clip = ImageSequenceClip(image_paths, fps=30)

            # Load the original video and extract its audio
            original_audio = AudioFileClip(tmp_file_path)

            # Set audio to the clip
            clip = clip.set_audio(original_audio)

            # Save the final video to a temporary file
            final_output_path = os.path.join(tmp_dir, "final_output.mp4")
            clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", fps=30)

            # Read the final video back into a BytesIO object
            with open(final_output_path, "rb") as f:
                final_output_memory_file = io.BytesIO(f.read())

            st.session_state.processed_video1 = final_output_memory_file
            progress_bar.progress(1.0, text="Processing complete!")

            with col2:
                st.video(final_output_memory_file, format="video/mp4")

            print(os.listdir(tmp_dir))

            # create another tmp dir for the generated video
            with tempfile.TemporaryDirectory() as gen_dir:
                print(f"Generated video directory: {gen_dir}")

                # start to test the model
                subprocess.run(
                    [
                        f"{sys.executable}",
                        "test_model.py",
                        "--dataroot",
                        f"{tmp_dir}",
                        "--results_dir",
                        f"{gen_dir}",
                        "--num_test",
                        f"{total_frames}",
                        "--gpu_ids",
                        "-1",
                    ]
                )

                # write the final video to the output
                image_gen_paths = natsorted([os.path.join(f"{gen_dir}", f"frame_{i}.png") for i in range(frame_number)])

                # Create video clip from image sequence
                clip = ImageSequenceClip(image_gen_paths, fps=30)

                # Load the original video and extract its audio
                original_audio = AudioFileClip(tmp_file_path)

                # Set audio to the clip
                clip = clip.set_audio(original_audio)

                # Save the final video to a temporary file
                gen_path = os.path.join(gen_dir, "video_gen.mp4")
                clip.write_videofile(gen_path, codec="libx264", audio_codec="aac", fps=30)

                # Read the final video back into a BytesIO object
                with open(gen_path, "rb") as f:
                    gen_memory_file = io.BytesIO(f.read())

                st.session_state.processed_video2 = gen_memory_file
                progress_bar.progress(1.0, text="Processing complete!")

                st.video(gen_memory_file, format="video/mp4")

        else:
            st.warning("未能讀取視頻幀")
