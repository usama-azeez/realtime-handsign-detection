# Realtime Handsign Detection

This project implements a real-time American Sign Language (ASL) hand sign recognition system using MediaPipe for hand landmark detection and a machine learning model (Random Forest) for classification. It includes both a desktop Python app and a web-based interface to predict hand signs from live video input and provides audio feedback.

---

## 📽️ Sample Demo Video

👉 [Click here to watch the demo on YouTube Part 1](https://youtu.be/c_3fOgwVXh4)  <br>
👉 [Click here to watch the demo on YouTube Part 2](https://youtu.be/E8wQjdtTmMw)


## Project Overview

- `collecting.py`: Capture hand sign images from webcam for dataset  
- `create_dataset.py`: Extract hand landmarks and prepare data for training  
- `training.py`: Train and save RandomForest model on processed dataset  
- `testing.py`: Realtime prediction using webcam, shows predicted letter with image & speech  
- `index.html`: Web interface for webcam capture and prediction display  
- `data/`: Folder containing collected images (not included due to size)  
- `model.p`: Trained model pickle file (should be placed here)  

---

> **Note:**  
> The app source code and dataset folders are **not included** in this repository due to their large size (over 1 GB).  
> If you need access to them or further assistance, please contact the author.

---

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- scikit-learn
- gTTS
- playsound
- numpy
- Flask (if using web backend)
  

## 🔍 Features

Real-time hand landmark detection using MediaPipe.

Dataset of 26 ASL alphabets with 100 images per sign.

Machine learning model training with scikit-learn’s Random Forest Classifier.

Model testing and evaluation via testing.py.

Integration with audio output for sign speech conversion.

Easy-to-use interface for capturing new data and testing model predictions.

---


## 🏁 How to Run

1. **Clone the Repository**
   git clone https://github.com/usama-azeez/realtime-handsign-detection.git
   cd realtime-handsign-detection

2. **Install Dependencies**
   pip install -r requirements.txt

## Usage

1. **Data Collection**
Run collecting.py to collect 100 images for each of the 26 classes:
   python collecting.py

2. **2. Dataset Preparation**
Run create_dataset.py to process images, extract landmarks, and save features:
   python create_dataset.py
This creates data.pickle containing features and labels.


3. **Model Training**
Train the Random Forest classifier on the prepared dataset:
   python training.py

4. **Real-Time Testing (Desktop App)**
Run testing.py to start webcam prediction with live overlay and audio pronunciation:
   python testing.py

5. **Web Interface**
Open index.html in a browser. Use Open Camera to start video capture. The page sends frames to a backend /predict endpoint for prediction and displays the result.

   **Note**: You need a Flask backend server running to handle predictions from the frontend (not included here).


**Important Notes**
The model expects 42 features per sample (21 hand landmarks * 2 coordinates).
Cooldown is applied to avoid repeating audio speech for the same detected sign.
Make sure your webcam is accessible and not used by other apps during capture.
The project assumes labels from 0 to 25 map to A-Z respectively.
Large files such as the app source code and dataset are omitted for repository size reasons.

## Credits 
Developed by **Usama Aziz**
University: University of Narowal
Date: Sep 2024
Email: usamaaziz304@gmail.com

Feel free to reach out if you have questions or want to contribute!
