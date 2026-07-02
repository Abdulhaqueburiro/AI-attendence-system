import cv2
import numpy as np
import os
import json

def train_recognizer():
    dataset_path = "dataset"
    face_samples = []
    labels = []
    label_map = {}  # numeric_label -> roll_no
    current_label = 0

    if not os.path.exists(dataset_path):
        print("ERROR: 'dataset' folder nahi mila. Pehle enrollment_helper.py chalao.")
        return

    student_folders = os.listdir(dataset_path)

    if len(student_folders) == 0:
        print("ERROR: Dataset folder khali hai. Koi student enroll nahi hua.")
        return

    for roll_no in student_folders:
        student_path = os.path.join(dataset_path, roll_no)
        if not os.path.isdir(student_path):
            continue

        label_map[current_label] = roll_no
        print(f"Loading images for {roll_no} (label {current_label})...")

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                face_samples.append(img)
                labels.append(current_label)

        current_label += 1

    if len(face_samples) == 0:
        print("ERROR: Koi valid image nahi mili training ke liye.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(labels))
    recognizer.save("trainer.yml")

    with open("labels.json", "w") as f:
        json.dump(label_map, f)

    print(f"Training complete! {len(student_folders)} students, {len(face_samples)} total images trained.")
    print("Files saved: trainer.yml, labels.json")

if __name__ == '__main__':
    train_recognizer()