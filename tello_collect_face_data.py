'''
Capture multiple faces from multiple users to be stored in a dataset directory.

==> Faces will be stored in the 'dataset/' directory (create it if it doesn't exist).
==> Each face will have a unique numeric ID as 1, 2, 3, etc.

Original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
Adapted by Marcelo Rovai - MJRoBot.org @ 21Feb18
Modified to use laptop webcam and auto-continue numbering by ChatGPT @ 2025
'''

import cv2
import os
import re

# Ask for the user ID
face_id = input('\n Enter user ID and press <return>: ')

# Initialize the webcam (0 is typically the default camera)
cam = cv2.VideoCapture(0)

# Set video width and height if needed (optional)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Make sure the dataset folder exists
dataset_dir = 'face_data'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Find the current highest count for this face_id
existing_files = [f for f in os.listdir(dataset_dir) if re.match(rf'User\.{face_id}\.\d+\.jpg', f)]
if existing_files:
    counts = [int(re.findall(rf'User\.{face_id}\.(\d+)\.jpg', f)[0]) for f in existing_files]
    start_count = max(counts) + 1
else:
    start_count = 1

print(f"\n [INFO] Found {len(existing_files)} existing samples for user {face_id}. Starting at {start_count}.")
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

count = start_count

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the captured face into the dataset folder
        cv2.imwrite(f"{dataset_dir}/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)
        print(f"Saved sample {count}")
        count += 1

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' to exit early
    if k == 27:
        break
    elif count >= start_count + 400:  # Collect 400 new samples per session
        break

print("\n [INFO] Exiting Program and cleaning up.")
cam.release()
cv2.destroyAllWindows()