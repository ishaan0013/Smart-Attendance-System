# Import Necessary modules
import os
import cv2
import dlib
import numpy as np
import pandas as pd
from datetime import datetime
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face recognition model from dlib
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat")


# Load the known faces from the 'known_faces' folder
known_faces_folder = 'known_faces'
known_faces_encodings = []

# Populate known_faces_encodings with the encodings of known faces
for person_folder in os.listdir(known_faces_folder):
    person_path = os.path.join(known_faces_folder, person_folder)

    if os.path.isdir(person_path):
        person_encodings = []

        # Iterate over each image in the person's folder
        for file_name in os.listdir(person_path):
            image = cv2.imread(os.path.join(person_path, file_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Assume there is only one face in the image (for simplicity)
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y + h, x:x + w]

                # Get face landmarks
                shape = shape_predictor(
                    image, dlib.rectangle(x, y, x + w, y + h))

                # Get face encoding
                face_encoding = face_recognizer.compute_face_descriptor(
                    image, shape)

                person_encodings.append({
                    'name': person_folder,
                    'encoding': face_encoding
                })

        # Add the encodings for the person to the main list
        known_faces_encodings.extend(person_encodings)


# Initialize webcam
cap = cv2.VideoCapture(0)

# Create CSV file for attendance records
csv_folder = 'attendance_records'
os.makedirs(csv_folder, exist_ok=True)

# Get the current date in the desired format
current_date_str = datetime.now().strftime("%d_%b_%Y")
csv_file = os.path.join(csv_folder, f"{current_date_str}.csv")

attendance_data = {'Name': [], 'Time': []}

next_person_notification = True


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Get face landmarks
        shape = shape_predictor(frame, dlib.rectangle(x, y, x + w, y + h))

        # Get face encoding
        face_encoding = face_recognizer.compute_face_descriptor(frame, shape)

        # Compare with the known faces
        for known_face in known_faces_encodings:
            distance = np.linalg.norm(
                np.array(known_face['encoding']) - np.array(face_encoding))

            # Threshold for face recognition
            if distance < 0.3:
                name = known_face['name']

                # Update attendance data
                if name not in attendance_data['Name']:
                    attendance_data['Name'].append(name)
                    attendance_data['Time'].append(
                        datetime.now().strftime("%H:%M:%S"))

                    # Display the recognized name on the frame
                    cv2.putText(
                        frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                    if next_person_notification:
                        # Speak "Next person" using text-to-speech
                        engine.say("Next person")
                        engine.runAndWait()
                        next_person_notification = False

                break

    # Reset notification flag when no faces are detected
    if len(faces) == 0:
        next_person_notification = True

    # Display the frame
    cv2.imshow('Smart Attendance System', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance data to CSV file
df = pd.DataFrame(attendance_data)
df.to_csv(csv_file, index=False)

# Release the webcam
cap.release()

# Release all OpenCV windows
cv2.destroyAllWindows()
