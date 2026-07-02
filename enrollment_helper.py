import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_face_samples(student_roll_no, num_samples=20):
    """
    Webcam se student ke chehre ke multiple samples capture karta hai
    aur 'dataset/<roll_no>/' folder mein save karta hai
    """
    save_path = f"dataset/{student_roll_no}"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Capturing samples for {student_roll_no}. Camera ke saamne dekho aur thora hilao apna sar.")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Enrollment - Capturing Face Samples', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! {count} samples saved for {student_roll_no}")
    return count

if __name__ == '__main__':
    roll_no = input("Student ka roll number likho: ")
    capture_face_samples(roll_no)