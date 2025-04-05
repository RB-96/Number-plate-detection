# 🚗 License Plate Detection & Recognition

An interactive Streamlit web app for detecting and recognizing vehicle license plates using a fine-tuned **YOLOv11** model and **PaddleOCR**. This solution supports both **image and video** inputs and allows for seamless demo usage, real-time preview, and CSV export of recognized plates.

---

## 🔧 Features

- ⚡ **YOLOv11** based detection (fine-tuned on public license plate datasets)
- 🧐 **PaddleOCR** for high-accuracy multilingual text recognition
- 📷 Supports both images and video files
- 📦 Easy-to-use **Streamlit UI**
- 📄 Export detected license plates to CSV
- 🎮 Includes demo mode for testing without uploading files

---

## 🧠 Model Info

- **Detection**: Custom-trained YOLOv8 (YOLOv11) model for car number plate localization
- **Recognition**: PaddleOCR (`use_angle_cls=True`, `lang='en'`)
- **Training Dataset**: Public datasets like https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
---

## 📁 Folder Structure

```
project_root/
├── yolo_applicaiton.py                  # Main Streamlit app
├── best_new.pt             # Fine-tuned YOLOv11 weights
├── demo/
│   └── WhatsApp Image 2025-04-04 at 15.14.09.jpeg  # Demo image
├── requirements.txt
└── ...
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/RB-96/Number-plate-detection.git
   cd Number-plate-detection
   ```

2. **Install dependencies**
   *(Python ≥ 3.8 recommended)*

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📸 Sample Demo

You can test the application without uploading anything by clicking the **"Use Demo File"** button in the sidebar.

> 📝 Make sure your demo image is in the `demo/` folder and named:
> `WhatsApp Image 2025-04-04 at 15.14.09.jpeg`

---

## 📄 Output Example

| Plate Text | Confidence (%) |
|------------|----------------|
| WB12A3456  | 95.2           |
| DL10AZ8765 | 91.7           |

You can download this result as a `.csv` from the app itself.

---

## 💡 Future Improvements

- Add multi-language OCR support
- Support for real-time webcam input
- Fine-tune detection for regional plate styles

---

## 👤 Author

Developed by **Raktima Barman**  
🔬 Data Scientist & Machine Learning Engineer  
📧 reddish.rb@gmail.com
📌 https://www.linkedin.com/in/raktimabarman96/

---

## 🛡️ License

This project is open-sourced under the [MIT License](LICENSE).

