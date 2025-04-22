from pyexpat import model

import cv2
import numpy as np
from app import preprocess_face


def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)) # type: ignore

    if len(faces) == 0:
        return frame

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    for (x, y, w, h) in faces[:1]:
        face_input = preprocess_face(gray, x, y, w, h)

        prediction = model.predict(face_input, verbose=0)[0]

        # 💡 Log the prediction for debugging
        print("Prediction probs:", dict(zip(class_labels, [round(p*100, 2) for p in prediction]))) # type: ignore

        max_index = np.argmax(prediction)
        confidence = prediction[max_index] * 100
        emotion_label = class_labels[max_index] # type: ignore

        # Skip confidence thresholding for testing — show actual prediction
        text = f"{emotion_label} ({confidence:.1f}%)"

        # 🔁 Also display top 3 predictions (optional)
        top3_indices = prediction.argsort()[-3:][::-1]
        top3 = [(class_labels[i], round(prediction[i]*100, 1)) for i in top3_indices] # type: ignore
        top_text = " | ".join([f"{lbl}: {conf}%" for lbl, conf in top3])

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, top_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return frame