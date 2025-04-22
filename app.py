from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "emotions.h5"
model = load_model(MODEL_PATH)

# Define class labels (adjust based on your dataset)
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Initialize OpenCV VideoCapture (0 = default webcam)
camera = cv2.VideoCapture(0)

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load OpenCV face detector

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract face
        face = cv2.resize(face, (48, 48))  # Resize to model input size
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize

        prediction = model.predict(face)  # Predict emotion
        emotion_label = class_labels[np.argmax(prediction)]  # Get label

        # Draw rectangle and emotion label on face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def generate_frames():
    while True:
        success, frame = camera.read()  # Read frame from webcam
        if not success:
            break

        frame = detect_emotion(frame)  # Process frame

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")  # Send frame

# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route for video feed
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
