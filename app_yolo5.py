# -*- coding: utf-8 -*-
import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Важно! Самый первый Streamlit-вызов
st.set_page_config(page_title="Детекция с дрона", layout="wide")
st.title("Детекция людей на фото, видео и с веб-камеры")

# Загрузка модели YOLOv5
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model


model = load_model()

option = st.radio("Выберите источник:", ["Фото", "Видео", "Веб-камера"])

# Фото
if option == "Фото":
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption='Оригинальное изображение', use_container_width=True)

        st.write("Обработка изображения...")
        results = model(image)

        st.image(np.squeeze(results.render()), caption="Результат детекции", use_container_width=True)

        df = results.pandas().xyxy[0]
        st.write("Обнаруженные объекты с координатами:")
        st.dataframe(df)

# Видео
elif option == "Видео":
    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        input_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        input_tmp.write(uploaded_video.read())
        input_tmp_path = input_tmp.name

        cap = cv2.VideoCapture(input_tmp_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(tempfile.gettempdir(), f"detected_{now}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st.write("Обработка видео...")
        frame_count = 0
        max_frames = 300

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > max_frames:
                break
            results = model(frame)
            annotated = np.squeeze(results.render())
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            frame_count += 1

        cap.release()
        out.release()

        st.success("✅ Обработка завершена")
        st.video(output_path)

        with open(output_path, "rb") as video_file:
            st.download_button("📥 Скачать обработанное видео", video_file, file_name=f"detected_{now}.mp4")

# Веб-камера
elif option == "Веб-камера":
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            rendered = np.squeeze(results.render())
            return cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

    st.write("Нажмите 'Start' для запуска веб-камеры")
    webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)
