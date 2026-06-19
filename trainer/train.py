import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Load face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Dataset folder
dataset_path = "dataset"

# Store face samples
face_samples = []

# Store IDs
ids = []

# Store labels
label_ids = {}

# Starting ID
current_id = 0

print("Training Started...")

# Read each person's folder
for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    # Skip files
    if not os.path.isdir(person_path):
        continue

    print("Processing:", person_name)

    # Assign ID
    label_ids[current_id] = person_name

    # Read images
    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        print("Reading:", image_path)

        try:
            # Convert image to grayscale
            PIL_img = Image.open(image_path).convert('L')

        except:
            print("Cannot open image:", image_name)
            continue

        # Convert image to numpy array
        img_numpy = np.array(PIL_img, 'uint8')

        # Detect faces
        faces = detector.detectMultiScale(img_numpy)

        # If no face found
        if len(faces) == 0:
            print("No face detected in", image_name)

        # Add face samples
        for (x, y, w, h) in faces:

            face_samples.append(
                img_numpy[y:y+h, x:x+w]
            )

            ids.append(current_id)

            print("Face Added:", image_name)

    current_id += 1

# Total faces
print("Total Faces Trained:", len(face_samples))

# Stop if no faces
if len(face_samples) == 0:
    print("No valid training data found")
    exit()

# Train recognizer
print("Training Faces...")

recognizer.train(face_samples, np.array(ids))

# Create trainer folder automatically
if not os.path.exists("trainer"):
    os.makedirs("trainer")

# Save trainer model
recognizer.write("trainer/trainer.yml")

# Save labels
with open("trainer/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("Training Completed Successfully")
print("trainer.yml created")
print("labels.pickle created")

