# Import Necessary modules
import cv2
import os
import time
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a folder for storing known faces which act as Database
known_faces_folder = 'known_faces'
os.makedirs(known_faces_folder, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Number of images to capture per person
images_per_person = 5

# Delay between each image capture (in seconds)
capture_delay = 2

# Loop to capture images for multiple persons
while True:
    # Prompt user to enter the name of the person
    person_name = input("Enter the name of the person (or type 'q' to quit): ")

    if person_name.lower() == 'q':
        break

    # Create a folder for the person if it doesn't exist
    person_folder = os.path.join(known_faces_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Capture images for the person
    capture_counter = 0
    while capture_counter < images_per_person:
        ret, frame = cap.read()

        # Detect faces in the frame and convert it to gray from BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If face found
        if len(faces) > 0:
            # Display the frame
            cv2.imshow('Capture Known Faces', frame)

            # Save the current frame as an image in the person's folder
            img_filename = f"{person_name}_{capture_counter + 1}.jpg"
            img_path = os.path.join(person_folder, img_filename)
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_filename}")

            # Increment the counter for successfully captured images
            capture_counter += 1

            # Wait for the specified delay before capturing the next image
            time.sleep(capture_delay)
        else:
            # If no face found, say "No face found"
            engine.say("No face found")
            engine.runAndWait()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
