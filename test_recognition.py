import cv2
import json

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.json", "r") as f:
    label_map = json.load(f)

# JSON keys hamesha string hoti hain, isliye conversion zaroori hai
label_map = {int(k): v for k, v in label_map.items()}

CONFIDENCE_THRESHOLD = 70  # Kam confidence = behtar match (LBPH mein distance hota hai)

cap = cv2.VideoCapture(0)
print("Recognition test shuru. 'q' dabake band karo.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)

        if confidence < CONFIDENCE_THRESHOLD:
            roll_no = label_map.get(label, "Unknown")
            text = f"{roll_no} ({round(confidence, 1)})"
            color = (0, 255, 0)
        else:
            text = f"Unknown ({round(confidence, 1)})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Recognition Test - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()