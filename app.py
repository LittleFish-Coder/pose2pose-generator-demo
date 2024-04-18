# import streamlit as st
# from PIL import Image

# # 初始化變數
# image_file = None
# video_file = None

# # 創建上傳圖片按鈕
# uploaded_image = st.file_uploader("選擇一張圖片", type=["jpg", "jpeg", "png"])
# if uploaded_image is not None:
#     image_file = Image.open(uploaded_image)

# # 創建上傳影片按鈕
# uploaded_video = st.file_uploader("選擇一個影片", type=["mp4", "avi", "mov"])
# if uploaded_video is not None:
#     video_file = uploaded_video.read()

# # 創建顯示按鈕
# if st.button("顯示圖片和影片"):
#     if image_file is not None:
#         st.image(image_file, caption="上傳的圖片")
#     else:
#         st.warning("請先上傳一張圖片。")

#     if video_file is not None:
#         st.video(video_file)
#     else:
#         st.warning("請先上傳一個影片文件。")
import streamlit as st
from PIL import Image
# import cv2
import tempfile

def process_pose_video(input_video_path):
    # 這裡添加處理影片的代碼，返回處理後的影片路徑
    return input_video_path  # 暫時回傳原路徑作為範例

def generate_video(input_video_path):
    # 這裡添加生成影片的代碼，返回生成的影片路徑
    return input_video_path  # 暫時回傳原路徑作為範例

# 畫面上半部
st.title("Video Processing App")

col1, col2, col3 = st.columns(3)
with col1:
    video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv'])

with col2:
    if st.button("Process Pose"):
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            processed_video_path = process_pose_video(tfile.name)
            st.video(processed_video_path)
        else:
            st.warning("Please upload a video file first.")

with col3:
    if video_file is not None:
        st.video(video_file)

# 畫面下半部
st.header("Additional Video Generation")

col4, col5 = st.columns(2)
with col4:
    if st.button("Generate"):
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            generated_video_path = generate_video(tfile.name)
            st.video(generated_video_path)
        else:
            st.warning("Please upload a video file first.")

with col5:
    pass  # 這裡可以展示生成後的影片或留白

