import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import torch
from paddleocr import PaddleOCR
import pandas as pd
import base64
import requests
from io import BytesIO
from pathlib import Path

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Page Config
st.set_page_config(page_title="License Plate Detection", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    uploaded_file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])
    use_demo = st.button("🎬 Use Demo File")

    st.markdown("---")
    st.info("🔍 Detection powered by YOLOv8 + PaddleOCR")

# Title
st.markdown("<h1 style='text-align: center;'>🚗 License Plate Detection & Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with YOLOv8 and PaddleOCR for accurate real-time recognition</p>", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best_new.pt")

try:
    model = load_model()
    st.success("✅ YOLO model loaded successfully.")
except Exception as e:
    st.error(f"🚨 Error loading YOLO model: {e}")

# OCR using PaddleOCR
def paddle_ocr(plate_roi):
    result = ocr.ocr(plate_roi, cls=True)
    texts = [line[1][0] for line in result[0]]
    return " ".join(texts).strip()

# Detection function
def detect_and_read(image):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = model.predict(image, device=device)

        image = np.array(image)
        plate_texts = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0]) * 100

                pad = 10
                h, w, _ = image.shape
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate_roi = image[y1:y2, x1:x2]

                if plate_roi is None or plate_roi.size == 0:
                    continue

                text = paddle_ocr(plate_roi)
                plate_texts.append((text.strip(), confidence))

                cv2.putText(image, f"{text} ({confidence:.1f}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return image, plate_texts

    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None, []
    
# Video processing logic
def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with st.spinner("⏳ Processing video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, _ = detect_and_read(frame_rgb)
            if processed_frame is not None:
                processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                out.write(processed_bgr)

    cap.release()
    out.release()
    return output_video_path

# CSV Export Helper
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Demo file (image or video)
def get_demo_image():
    demo_dir = Path.cwd() / "demo"
    demo_path = demo_dir / "WhatsApp Image 2025-04-04 at 15.14.09.jpeg"
    
    if not demo_path.exists():
        st.error(f"Demo image not found at: {demo_path}")
        return None
    
    return Image.open(demo_path).convert("RGB")

# Handle input
if uploaded_file or use_demo:
    is_demo = use_demo and not uploaded_file
    file_ext = None

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name

    if is_demo:
        demo_img = get_demo_image()
        if demo_img:
            st.markdown("### 🧪 Using demo image")
            st.image(demo_img, caption="Demo Image", use_column_width=False)
            st.markdown("### 🔍 Detecting...")
            processed_img, texts = detect_and_read(demo_img)
    elif file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image = Image.open(file_path).convert("RGB")
        # st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        st.markdown("### 🔍 Detecting...")
        processed_img, texts = detect_and_read(image)
    elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            output_path = output_file.name
        st.markdown("### 🎞️ Processing video...")
        result = process_video(file_path, output_path)
        if result:
            st.video(result)
        texts = []
    else:
        st.error("❌ Unsupported file format.")
        texts = []

    # Show result
    if texts:
        st.image(processed_img, caption="✅ Processed Image", use_column_width=False)
        df = pd.DataFrame(texts, columns=["Plate Text", "Confidence (%)"])
        st.markdown("### 🧾 Detected License Plates")
        st.dataframe(df, use_container_width=True)

        # csv = convert_df_to_csv(df)
        # st.download_button(
        #     label="📥 Download as CSV",
        #     data=csv,
        #     file_name="detected_plates.csv",
        #     mime="text/csv"
        # )
