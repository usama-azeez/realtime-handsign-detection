from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
import os
from gtts import gTTS
from playsound import playsound
import threading
import time

app = Flask(__name__)
CORS(app)

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary for the signs
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Dictionary to store the paths for images
image_paths_dict = {value: f'data/{key}/0.jpg' for key, value in labels_dict.items()}

# Function to speak the predicted character
def speak_character(character):
    def _speak():
        tts = gTTS(text=character, lang='en')
        tts.save("temp.mp3")
        playsound("temp.mp3")
        os.remove("temp.mp3")

    threading.Thread(target=_speak).start()

# Function to predict gesture from an uploaded image
def predict_gesture(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    predictions = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        if len(data_aux) == 42:  # Assuming 21 landmarks, each with x and y
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            predictions.append(predicted_character)

            # Speak the predicted character
            speak_character(predicted_character)

    return predictions

# Endpoint to predict gesture from uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)

    # Process the image file
    img = cv2.imread(temp_file.name)
    if img is None:
        os.remove(temp_file.name)
        return jsonify({'error': 'Invalid image file'}), 400

    # Predict gesture from the uploaded image
    predictions = predict_gesture(img)

    # Get the image path for the predicted character
    image_path = None
    if predictions:
        predicted_label = predictions[0]
        image_path = image_paths_dict.get(predicted_label)

    os.remove(temp_file.name)

    # Return response with the predicted character and image path
    return jsonify({'predictions': predictions, 'image_path': image_path})

# Endpoint to serve images based on label (optional)
@app.route('/image/<label>', methods=['GET'])
def get_image(label):
    try:
        label = label.upper()
        image_path = image_paths_dict[label]
        return send_file(image_path, mimetype='image/jpeg')
    except KeyError:
        return jsonify({'error': 'Invalid label'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
