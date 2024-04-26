import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from natsort import natsorted
import tempfile

def GAN_model(video):
    # 將pose estimation後的圖片做GAN model
    return video

def main():
    st.title("YYDS影片生成器")

    # 輸入影片
    uploaded_file = st.file_uploader("選擇一個影片檔", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.video(uploaded_file)

    pose_estimation_button = st.button("pose_estimation", key="pose_estimation")
    if pose_estimation_button and uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Initialize MediaPipe Pose and Drawing utilities
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            pose = mp_pose.Pose()

            # Open the video file
            cap = cv2.VideoCapture(tmp_file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)  # 初始化進度條

            # Initialize an empty frame and black background image
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                black_background = np.zeros_like(frame_rgb)
            else:
                print("Failed to read the first frame from the video file.")
                cap.release()
                exit()

            frame_number = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create a black background image
                black_background = np.zeros_like(frame_rgb)

                # Process the frame with MediaPipe Pose
                result = pose.process(frame_rgb)

                # Draw the pose landmarks on the black background
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(black_background, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                image_path = os.path.join(tmp_dir, f'frame_{frame_number}.png')
                cv2.imwrite(image_path, black_background)
                print(f"成功保存圖片: {image_path}")
                # 每當 frame_number 可被 10 整除時，顯示該幀的圖片
                # if frame_number % 10 == 0:
                #     st.image(black_background, caption=f"Frame {frame_number}", use_column_width=True)
                frame_number += 1
                progress = frame_number / total_frames
                progress_bar.progress(progress)

            cap.release()
            progress = total_frames / total_frames
            progress_bar.progress(progress)
            print("pose_estimation完成")
            video_placeholder = st.empty()
            # 檢查 frame_number 是否大於 0
            if frame_number > 0:
                # image_paths = os.path.join(tmp_dir)
                # print(f"image_paths: {image_paths}")  # 打印 image_paths
                image_paths = natsorted(os.path.join(tmp_dir, f'frame_{frame_num}.png') for frame_num in range(frame_number))
                
                # 建立影片編碼器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                frame_size = (black_background.shape[1], black_background.shape[0])
                # print(f"影片路徑: {os.path.join(tmp_dir, 'output_video.mp4')}")
                out = cv2.VideoWriter(os.path.join(tmp_dir, 'output_video.mp4'), fourcc, fps, frame_size)

                # 寫入影格並編碼成影片
                # for image_path in image_paths:
                #     frame = cv2.imread(image_path)
                #     out.write(frame)
                for image_path in image_paths:
                    frame = cv2.imread(image_path)
                    out.write(frame)
                    print(f"寫入影格: {image_path}")  # 打印寫入的影格路徑
                # 釋放影片編碼器
                out.release()
                print("影片生成完成")
                # video_file = open(os.path.join(tmp_dir, 'output_video.mp4'), 'rb')
                # video_bytes = video_file.read()
                # st.video(video_bytes)
                st.session_state.processed_video1 = video_placeholder.video(os.path.join(tmp_dir, 'output_video.mp4'))
                
                print(f"影片路徑: {os.path.join(tmp_dir, 'output_video.mp4')}")
            else:
                st.warning("未能讀取視頻幀")
    if 'processed_video1' in st.session_state:
        st.video(st.session_state.processed_video1)

   
    generate_button = st.button("generate", key="generate")
    if generate_button and uploaded_file is not None:
        st.session_state.processed_video2 = GAN_model(uploaded_file)
    if 'processed_video2' in st.session_state:
        st.video(st.session_state.processed_video2)

if __name__ == "__main__":
    main()