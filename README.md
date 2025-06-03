# realtime-handsign-detection
Real-time ASL sign detection using Python, MediaPipe, and machine learning.

# Real-Time Sign Language Detection

This project is a Final Year Project (FYP) designed to aid communication through real-time American Sign Language (ASL) detection using computer vision and machine learning.

---

## 🚀 Project Overview

This system detects and translates American Sign Language (ASL) alphabets in real-time using a webcam. It was developed to bridge communication gaps for the deaf and hard-of-hearing community.

## 🔍 Features

- Real-time ASL alphabet recognition (A-Z)
- Live webcam integration
- Hand landmark detection using MediaPipe
- Machine learning with scikit-learn (Random Forest Classifier)
- Dataset collection and model training scripts included

---

## 🧠 Tech Stack

- **Language**: Python
- **Libraries**: OpenCV, MediaPipe, scikit-learn, pickle

---

## 📁 Project Structure

📦 project/
├── app.py # Real-time detection app
├── create_dataset.py # Script to collect dataset images
├── training.py # Train the ML model
├── testing.py # Evaluate/test model
├── model.p # Trained model file
├── data.pickle # Pickled data labels
├── index.html # Optional web interface
├── Video.mov # Demo video
└── data/ # Folder containing dataset folders from 0-25 (A-Z)


> Note: Dataset contains 26 folders (0 to 25), each with 100 labeled images representing ASL signs A-Z.

---

## 🏁 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/realtime-handsign-detection.git
   cd realtime-handsign-detection

2. **Install Dependencies**
3. **Collect Dataset (Optional)**
4. **Train Model**
5. **Run Testing**



