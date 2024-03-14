import streamlit as st
from PIL import Image
import cv2

# 初始化變數
image_file = None
video_file = None

# 創建上傳圖片按鈕
uploaded_image = st.file_uploader("選擇一張圖片", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image_file = Image.open(uploaded_image)

# 創建上傳影片按鈕
uploaded_video = st.file_uploader("選擇一個影片", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    video_file = uploaded_video.read()

# 創建顯示按鈕
if st.button("顯示圖片和影片"):
    if image_file is not None:
        st.image(image_file, caption="上傳的圖片")
    else:
        st.warning("請先上傳一張圖片。")

    if video_file is not None:
        st.video(video_file)
    else:
        st.warning("請先上傳一個影片文件。")