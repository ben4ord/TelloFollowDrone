import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'tellodataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # Load image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract user ID from the filename
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            # Extract the face region
            face = img_numpy[y:y+h, x:x+w]

            # Resize the face to 200x200 for consistency
            face_resized = cv2.resize(face, (200, 200))
            faceSamples.append(face_resized)
            ids.append(id)

            # Simulate smaller faces by downscaling and then upscaling
            for scale_factor in [0.5, 0.3]:
                small_face = cv2.resize(face_resized, (0, 0), fx=scale_factor, fy=scale_factor)
                simulated_face = cv2.resize(small_face, (200, 200))
                faceSamples.append(simulated_face)
                ids.append(id)

    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.save('tellotrainer.yml')

# Print the number of faces trained and end program
print(f"\n [INFO] {len(np.unique(ids))} unique users trained. Total samples: {len(faces)}. Exiting Program.")
