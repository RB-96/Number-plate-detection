import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import easyocr

# Set Streamlit title
st.title("Number Plate Detection & Recognition using YOLO")

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load YOLO model
try:
    model = YOLO('best_new.pt')  # Replace with the path to your trained YOLO model
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# Load OCR model
reader = easyocr.Reader(['en'])

def detect_and_read(image):
    try:
        results = model.predict(image, device="gpu")
        image = np.array(image)
        plate_texts = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract ROI for OCR
                plate_roi = image[y1:y2, x1:x2]
                plate_text = reader.readtext(plate_roi, detail=0)
                plate_texts.append(" ".join(plate_text))
                
                # Display OCR result
                text = plate_texts[-1] if plate_texts else "Unknown"
                cv2.putText(image, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return image, plate_texts
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, []

def process_video(video_path, output_video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, plate_texts = detect_and_read(frame_rgb)
            
            out.write(processed_frame)
        
        cap.release()
        out.release()
        return output_video_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

# Handle uploaded file
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("Processing...")

    if file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        image = Image.open(temp_file_path)
        processed_image, plate_texts = detect_and_read(image)
        if processed_image is not None:
            st.image(processed_image)
            st.write("Detected License Plates:", plate_texts)
    elif file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            output_video_path = output_file.name

        result_path = process_video(temp_file_path, output_video_path)
        
        if result_path:
            st.video(result_path)
    else:
        st.error("Unsupported file format!")
