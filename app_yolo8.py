# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Детекция с дрона", layout="wide")
st.title("Детекция людей на фото, видео и с веб-камеры")

@st.cache_resource
def load_model():
    model = YOLO('yolov8s.pt')
    model.eval()
    return model

model = load_model()

def filter_person_boxes(results):
    person_class_id = 0
    boxes = results[0].boxes
    # Фильтруем только класс "person"
    person_boxes = boxes[boxes.cls == person_class_id]
    return person_boxes

def plot_filtered_results(results):
    person_boxes = filter_person_boxes(results)
    image = results[0].orig_img.copy()

    for box in person_boxes:
        x1, y1, x2, y2, conf, cls = box.data.cpu().numpy()[0]
        # Рисуем только "person"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image

def convert_to_dataframe(boxes):
    return pd.DataFrame(
        boxes.data.cpu().numpy(),
        columns=["x1", "y1", "x2", "y2", "confidence", "class"]
    )

option = st.radio("Выберите источник:", ["Фото", "Видео", "Веб-камера"])

if option == "Фото":
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption='Оригинальное изображение', use_container_width=True)

        results = model(image)
        filtered_image = plot_filtered_results(results)
        st.image(filtered_image, caption="Только люди", use_container_width=True)

        # Вывод данных в таблицу
        person_boxes = filter_person_boxes(results)
        df = convert_to_dataframe(person_boxes)
        st.write("Обнаруженные объекты класса 'person':")
        st.dataframe(df)

elif option == "Видео":
    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi"])
    if uploaded_video:
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            filtered_frame = plot_filtered_results(results)
            out.write(filtered_frame)

        cap.release()
        out.release()

        st.video(output_path)

        with open(output_path, "rb") as video_file:
            st.download_button("Скачать видео", video_file, file_name=f"detected_{now}.mp4")

elif option == "Веб-камера":
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            filtered_frame = plot_filtered_results(results)
            return filtered_frame

    webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)
