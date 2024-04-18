import streamlit as st
import cv2
import numpy as np
import tempfile

def pose_estimation(video):
    # 將input影片做pose estimation
    return video

def GAN_model(video):
    # 將pose estimation後的圖片做GAN model
    return video

def main():
    st.title("姿勢估計")

    # 輸入影片
    uploaded_file = st.file_uploader("選擇一個影片檔", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video = cv2.VideoCapture(tmp_file.name)
            st.video(uploaded_file)

    cols = st.columns(2)
    cols2 = st.columns(2)

    with cols[0]:
        pose_estimation_button = st.button("pose_estimation", key="pose_estimation")
        if pose_estimation_button and uploaded_file is not None:
            st.session_state.processed_video1 = pose_estimation(uploaded_file)
        if 'processed_video1' in st.session_state:
            st.video(st.session_state.processed_video1)

    with cols2[0]:
        generate_button = st.button("generate", key="generate")
        if generate_button and uploaded_file is not None:
            st.session_state.processed_video2 = GAN_model(uploaded_file)
        if 'processed_video2' in st.session_state:
             st.video(st.session_state.processed_video2)

if __name__ == "__main__":
    main()