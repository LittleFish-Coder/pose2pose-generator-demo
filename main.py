import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from natsort import natsorted
import tempfile
import io
import av
st.set_page_config(layout="wide")
st.title("YYDS影片生成器")
# set wide mode
col1, col2, col3 = st.columns([3, 1, 3],gap='large')

with col1:
# 輸入影片
    uploaded_file = st.file_uploader("選擇一個影片檔", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.video(uploaded_file)
with col2:
    st.write('')
    inference_button = st.button('Inference')
with col3:
    if inference_button:
        st.write('Inference button clicked')
        st.write('Displaying the image')
        if uploaded_file is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Initialize MediaPipe Pose and Drawing utilities
                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                pose = mp_pose.Pose()
                mp_drawing_styles = mp.solutions.drawing_styles

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
                mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法
                mp_hands = mp.solutions.hands
                # mediapipe 啟用偵測全身
                # with mp_holistic.Holistic(
                #     min_detection_confidence=0.1,
                #     min_tracking_confidence=0.1) as holistic:
                #     while cap.isOpened():
                    
                #         ret, frame = cap.read()
                #         if not ret:
                #             break

                #         # Convert the frame to RGB
                #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #         # Create a black background image
                #         black_background = np.zeros_like(frame_rgb)

                #         # Process the frame with MediaPipe Pose
                #         # results = pose.process(frame_rgb)
                #         result = holistic.process(frame_rgb)              # 開始偵測全身
                #         # Draw the pose landmarks on the black background
                #         if result.pose_landmarks:
                #             mp_drawing.draw_landmarks(black_background, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                #          # 身體偵測，繪製身體骨架
                #         mp_drawing.draw_landmarks(
                #             black_background,
                #             result.pose_landmarks,
                #             mp_holistic.POSE_CONNECTIONS,
                #             landmark_drawing_spec=mp_drawing_styles
                #             .get_default_pose_landmarks_style())
                #         image_path = os.path.join(tmp_dir, f'frame_{frame_number}.png')
                #         cv2.imwrite(image_path, black_background)
                with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
                    while cap.isOpened():
                        
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Convert the frame to RGB
                        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        # Create a black background image
                        black_background = np.zeros_like(frame)
                        # Read an image, flip it around y-axis for correct handedness output (see
                        
                        mp_drawing.draw_landmarks(
                            black_background,
                            results.hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                            # if frame_number % 10 == 0:
                            #     st.image(black_background, caption=f"Frame {frame_number}", use_column_width=True)
                        image_path = os.path.join(tmp_dir, f'frame_{frame_number}.png')
                        cv2.imwrite(image_path, black_background)
                        # Draw hand world landmarks.
                        if not results.multi_hand_world_landmarks:
                            continue
                        for hand_world_landmarks in results.multi_hand_world_landmarks:
                            mp_drawing.plot_landmarks(
                                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
                        frame_number += 1
                        progress = frame_number / total_frames
                        progress_bar.progress(progress)
                    
                # while cap.isOpened():
                    
                #     ret, frame = cap.read()
                #     if not ret:
                #         break

                #     # Convert the frame to RGB
                #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #     # Create a black background image
                #     black_background = np.zeros_like(frame_rgb)

                #     # Process the frame with MediaPipe Pose
                #     result = pose.process(frame_rgb)

                #     # Draw the pose landmarks on the black background
                #     if result.pose_landmarks:
                #         mp_drawing.draw_landmarks(black_background, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
                #     image_path = os.path.join(tmp_dir, f'frame_{frame_number}.png')
                #     cv2.imwrite(image_path, black_background)

                #     # if frame_number % 10 == 0:
                #     #     st.image(black_background, caption=f"Frame {frame_number}", use_column_width=True)
                #     frame_number += 1
                #     progress = frame_number / total_frames
                #     progress_bar.progress(progress)

                cap.release()
                video_placeholder = st.empty()
                # 檢查 frame_number 是否大於 0
                if frame_number > 0:
                    image_paths = natsorted(os.path.join(tmp_dir, f'frame_{frame_num}.png') for frame_num in range(frame_number))
                    n_frames = len(image_paths)
                    width, height, fps = black_background.shape[1], black_background.shape[0], 30
                    output_memory_file = io.BytesIO()
                    output = av.open(output_memory_file, 'w',format='mp4')
                    stream = output.add_stream('h264', rate=fps)
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = 'yuv420p'
                    for image_path in image_paths:
                        frame = cv2.imread(image_path)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                        packet = stream.encode(frame)
                        output.mux(packet)
                    packet = stream.encode(None)
                    output.mux(packet)
                    output.close()
                    
                    print("pose_estimation完成")
                    output_memory_file.seek(0)
                    st.session_state.processed_video1 = output_memory_file
                    # video_placeholder.video(output_memory_file, format='video/mp4')
                    progress = frame_number / frame_number
                    progress_bar.progress(progress)
                    if 'processed_video1' in st.session_state:
                        video_placeholder.video(st.session_state.processed_video1,format='video/mp4')
                    
                else:
                    st.warning("未能讀取視頻幀")


# generate_button = st.button("generate", key="generate")
# if generate_button and uploaded_file is not None:
#     st.session_state.processed_video2 = GAN_model(uploaded_file)
# if 'processed_video2' in st.session_state:
#     st.video(st.session_state.processed_video2)