from flask import Flask, request, jsonify, render_template
from db_config import get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import cv2
import numpy as np
import json
import base64
import math
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key-later-123456'

# ---------------- Face Recognition setup ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

CONFIDENCE_THRESHOLD = 70

# ---------------- Liveness (blink) setup ----------------
BaseOptions = mp_python.BaseOptions
FaceLandmarker = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(landmarker_options)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.21


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    vertical1 = euclidean(coords[1], coords[5])
    vertical2 = euclidean(coords[2], coords[4])
    horizontal = euclidean(coords[0], coords[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def decode_base64_image(img_b64):
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]
    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def get_ear_sequence(frames_b64):
    """Har frame se average EAR nikalta hai (jahan face landmarks mil jayen)."""
    ear_values = []
    for img_b64 in frames_b64:
        frame = decode_base64_image(img_b64)
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            h, w, _ = frame.shape
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear_values.append((left_ear + right_ear) / 2.0)
    return ear_values


def detect_blink_in_sequence(ear_values):
    """True agar EAR sequence mein 'band -> khuli' wala dip-and-rise pattern mile."""
    went_below = False
    for ear in ear_values:
        if ear < EAR_THRESHOLD:
            went_below = True
        elif went_below and ear >= EAR_THRESHOLD:
            return True
    return False


def retrain_model():
    """Dataset folder se model retrain karta hai aur running app mein hot-reload karta hai
    (server restart ki zarurat nahi)."""
    global recognizer, label_map

    dataset_path = "dataset"
    face_samples = []
    labels = []
    new_label_map = {}
    current_label = 0

    student_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    for roll in student_folders:
        student_path = os.path.join(dataset_path, roll)
        new_label_map[current_label] = roll
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                face_samples.append(img)
                labels.append(current_label)
        current_label += 1

    new_recognizer = cv2.face.LBPHFaceRecognizer_create()
    new_recognizer.train(face_samples, np.array(labels))
    new_recognizer.save("trainer.yml")

    with open("labels.json", "w") as f:
        json.dump(new_label_map, f)

    recognizer = new_recognizer
    label_map = new_label_map


# ---------------- Routes ----------------

@app.route('/chatbot/ask', methods=['POST'])
def chatbot_ask():
    data = request.get_json()
    message = data.get('message')
    role = data.get('role')
    user_id = data.get('user_id')

    if not all([message, role, user_id]):
        return jsonify({"error": "message, role, and user_id are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        if role == 'student':
            cursor.execute("SELECT name, roll_no FROM students WHERE id = %s", (user_id,))
            student = cursor.fetchone()
            if not student:
                cursor.close(); conn.close()
                return jsonify({"error": "Student not found"}), 404

            cursor.execute("""
                SELECT courses.id AS course_id, courses.course_name AS course_name,
                       courses.course_code AS course_code, courses.total_lectures AS total_lectures
                FROM enrollments
                JOIN courses ON enrollments.course_id = courses.id
                WHERE enrollments.student_id = %s
            """, (user_id,))
            courses = cursor.fetchall()

            context_lines = []
            for course in courses:
                total_lectures = course['total_lectures'] or 45
                cursor.execute("SELECT COUNT(*) AS conducted FROM lectures WHERE course_id = %s", (course['course_id'],))
                conducted = cursor.fetchone()['conducted']
                cursor.execute("""
                    SELECT COUNT(*) AS attended FROM attendance
                    JOIN lectures ON attendance.lecture_id = lectures.id
                    WHERE lectures.course_id = %s AND attendance.student_id = %s
                """, (course['course_id'], user_id))
                attended = cursor.fetchone()['attended']
                percentage = round((attended / total_lectures) * 100, 1) if total_lectures > 0 else 0
                status = "Safe" if percentage >= 75 else ("Condonation" if percentage >= 65 else "Not Eligible")
                context_lines.append(
                    f"- {course['course_name']} ({course['course_code']}): {attended}/{total_lectures} attended "
                    f"({conducted} lectures conducted so far), {percentage}% — Status: {status}"
                )

            cursor.execute("""
                SELECT notifications.message AS message, courses.course_name AS course_name
                FROM notifications
                JOIN courses ON notifications.course_id = courses.id
                WHERE notifications.student_id = %s
                ORDER BY notifications.created_at DESC LIMIT 5
            """, (user_id,))
            notifs = cursor.fetchall()
            notif_lines = [f"- [{n['course_name']}] {n['message']}" for n in notifs]

            system_prompt = (
                f"You are a friendly attendance assistant for {student['name']} (Roll No: {student['roll_no']}) "
                f"at the CS Department, QUEST Nawabshah. Answer their questions about attendance using ONLY "
                f"the data below. Be concise and conversational (2-4 sentences). If something isn't in the data, "
                f"say you don't have that information rather than guessing.\n\n"
                f"Attendance Policy: 75%+ is Safe, 65-74% is Condonation, below 65% is Not Eligible for exams.\n\n"
                f"Their course attendance:\n" + ("\n".join(context_lines) if context_lines else "No enrolled courses.") +
                f"\n\nRecent notifications:\n" + ("\n".join(notif_lines) if notif_lines else "No notifications.")
            )

        elif role == 'teacher':
            cursor.execute("SELECT name FROM teachers WHERE id = %s", (user_id,))
            teacher = cursor.fetchone()
            if not teacher:
                cursor.close(); conn.close()
                return jsonify({"error": "Teacher not found"}), 404

            cursor.execute("SELECT id, course_name, course_code, total_lectures FROM courses WHERE teacher_id = %s", (user_id,))
            courses = cursor.fetchall()

            context_lines = []
            for course in courses:
                cursor.execute("SELECT COUNT(*) AS conducted FROM lectures WHERE course_id = %s", (course['id'],))
                conducted = cursor.fetchone()['conducted']
                cursor.execute("SELECT COUNT(DISTINCT student_id) AS enrolled FROM enrollments WHERE course_id = %s", (course['id'],))
                enrolled = cursor.fetchone()['enrolled']
                context_lines.append(
                    f"- {course['course_name']} ({course['course_code']}): {enrolled} students enrolled, "
                    f"{conducted} lectures conducted so far out of {course['total_lectures']} planned."
                )

            system_prompt = (
                f"You are a helpful assistant for {teacher['name']}, a teacher at the CS Department, QUEST Nawabshah. "
                f"Answer their questions using ONLY the data below. Be concise and conversational (2-4 sentences).\n\n"
                f"Their assigned courses:\n" + ("\n".join(context_lines) if context_lines else "No courses assigned yet.")
            )
        else:
            cursor.close(); conn.close()
            return jsonify({"error": "Invalid role"}), 400

        cursor.close()
        conn.close()

        response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": message}]
        )

        answer = response.content[0].text
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

@app.route('/teacher/login')
def teacher_login_page():
    return render_template('teacher_login.html')

@app.route('/teacher/register')
def teacher_register_page():
    return render_template('teacher_register.html')

@app.route('/teacher/dashboard')
def teacher_dashboard_page():
    return render_template('teacher_dashboard.html')

@app.route('/admin/dashboard')
def admin_dashboard_page():
    return render_template('admin_dashboard.html')

@app.route('/student/login')
def student_login_page():
    return render_template('student_login.html')

@app.route('/student/register')
def student_register_page():
    return render_template('student_register.html')

@app.route('/student/enroll')
def student_enroll_page():
    return render_template('student_enroll.html')

@app.route('/student/dashboard')
def student_dashboard_page():
    return render_template('student_dashboard.html')

@app.route('/test-db')
def test_db():
    try:
        conn = get_db_connection()
        if conn.is_connected():
            conn.close()
            return "Database Connected Successfully!"
    except Exception as e:
        return f"Database Connection Failed: {str(e)}"

@app.route('/register/student', methods=['POST'])
def register_student():
    data = request.get_json()
    name = data.get('name')
    roll_no = data.get('roll_no')
    email = data.get('email')
    password = data.get('password')
    department = data.get('department')

    if not all([name, roll_no, email, password]):
        return jsonify({"error": "Missing required fields"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, roll_no, email, password, department) VALUES (%s, %s, %s, %s, %s)",
            (name, roll_no, email, hashed_password, department)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Student registered successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register/teacher', methods=['POST'])
def register_teacher():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    department = data.get('department')

    if not all([name, email, password]):
        return jsonify({"error": "Missing required fields"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO teachers (name, email, password, department) VALUES (%s, %s, %s, %s)",
            (name, email, hashed_password, department)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Teacher registered successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login/student', methods=['POST'])
def login_student():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"error": "Email and password required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE email = %s", (email,))
        student = cursor.fetchone()
        cursor.close()
        conn.close()

        if not student or not check_password_hash(student['password'], password):
            return jsonify({"error": "Invalid email or password"}), 401

        token = jwt.encode({
            'id': student['id'],
            'role': 'student',
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            "message": "Login successful",
            "token": token,
            "student": {
                "id": student['id'],
                "name": student['name'],
                "roll_no": student['roll_no'],
                "email": student['email']
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login/teacher', methods=['POST'])
def login_teacher():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"error": "Email and password required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM teachers WHERE email = %s", (email,))
        teacher = cursor.fetchone()
        cursor.close()
        conn.close()

        if not teacher or not check_password_hash(teacher['password'], password):
            return jsonify({"error": "Invalid email or password"}), 401

        token = jwt.encode({
            'id': teacher['id'],
            'role': teacher.get('role', 'teacher'),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            "message": "Login successful",
            "token": token,
            "teacher": {
                "id": teacher['id'],
                "name": teacher['name'],
                "email": teacher['email'],
                "role": teacher.get('role', 'teacher')
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/courses/add', methods=['POST'])
def add_course():
    data = request.get_json()
    course_name = data.get('course_name')
    course_code = data.get('course_code')
    teacher_id = data.get('teacher_id')
    semester = data.get('semester', 'Current Semester')
    total_lectures = data.get('total_lectures', 45)

    if not all([course_name, course_code, teacher_id]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO courses (course_name, course_code, teacher_id, semester, total_lectures) VALUES (%s, %s, %s, %s, %s)",
            (course_name, course_code, teacher_id, semester, total_lectures)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Course added successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/teachers/all', methods=['GET'])
def get_all_teachers():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, email FROM teachers WHERE role = 'teacher'")
        teachers = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"teachers": teachers}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/courses/all/detailed', methods=['GET'])
def get_all_courses_detailed():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT courses.id AS id, courses.course_name AS course_name, courses.course_code AS course_code,
                   courses.semester AS semester, courses.total_lectures AS total_lectures,
                   courses.teacher_id AS teacher_id, teachers.name AS teacher_name
            FROM courses
            LEFT JOIN teachers ON courses.teacher_id = teachers.id
            ORDER BY courses.id DESC
        """)
        courses = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"courses": courses}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/courses/reassign', methods=['POST'])
def reassign_course():
    data = request.get_json()
    course_id = data.get('course_id')
    teacher_id = data.get('teacher_id')

    if not all([course_id, teacher_id]):
        return jsonify({"error": "course_id and teacher_id are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE courses SET teacher_id = %s WHERE id = %s", (teacher_id, course_id))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Course reassigned successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/courses/teacher/<int:teacher_id>', methods=['GET'])
def get_teacher_courses(teacher_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM courses WHERE teacher_id = %s", (teacher_id,))
        courses = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"courses": courses}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/lectures/start', methods=['POST'])
def start_lecture():
    data = request.get_json()
    course_id = data.get('course_id')

    if not course_id:
        return jsonify({"error": "course_id is required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        today = datetime.date.today()
        now = datetime.datetime.now().time()

        cursor.execute(
            "INSERT INTO lectures (course_id, lecture_date, start_time, status) VALUES (%s, %s, %s, %s)",
            (course_id, today, now, 'active')
        )
        conn.commit()
        lecture_id = cursor.lastrowid
        cursor.close()
        conn.close()

        return jsonify({
            "message": "Lecture started successfully!",
            "lecture_id": lecture_id
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/attendance/lecture/<int:lecture_id>', methods=['GET'])
def get_lecture_attendance(lecture_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT students.name AS name, students.roll_no AS roll_no, attendance.marked_at AS marked_at
            FROM attendance
            JOIN students ON attendance.student_id = students.id
            WHERE attendance.lecture_id = %s
            ORDER BY attendance.marked_at DESC
        """, (lecture_id,))
        records = cursor.fetchall()
        cursor.close()
        conn.close()

        for r in records:
            if r.get('marked_at'):
                r['marked_at'] = r['marked_at'].strftime('%I:%M %p')

        return jsonify({"count": len(records), "students": records}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/attendance/student/<int:student_id>', methods=['GET'])
def get_student_attendance(student_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT courses.id AS course_id, courses.course_name AS course_name, courses.course_code AS course_code,
                   courses.total_lectures AS total_lectures
            FROM enrollments
            JOIN courses ON enrollments.course_id = courses.id
            WHERE enrollments.student_id = %s
        """, (student_id,))
        courses = cursor.fetchall()

        result = []
        for course in courses:
            total_lectures = course['total_lectures'] or 45

            cursor.execute("""
                SELECT COUNT(*) AS conducted
                FROM lectures
                WHERE course_id = %s
            """, (course['course_id'],))
            conducted = cursor.fetchone()['conducted']

            cursor.execute("""
                SELECT COUNT(*) AS attended
                FROM attendance
                JOIN lectures ON attendance.lecture_id = lectures.id
                WHERE lectures.course_id = %s AND attendance.student_id = %s
            """, (course['course_id'], student_id))
            attended = cursor.fetchone()['attended']

            percentage = round((attended / total_lectures) * 100, 1) if total_lectures > 0 else 0

            if percentage >= 75:
                eligibility = "safe"
            elif percentage >= 65:
                eligibility = "condonation"
            else:
                eligibility = "not_eligible"

            result.append({
                "course_name": course['course_name'],
                "course_code": course['course_code'],
                "total_lectures": total_lectures,
                "conducted": conducted,
                "attended": attended,
                "percentage": percentage,
                "eligibility": eligibility
            })

        cursor.execute("""
            SELECT lectures.lecture_date AS lecture_date, attendance.marked_at AS marked_at,
                   courses.course_name AS course_name
            FROM attendance
            JOIN lectures ON attendance.lecture_id = lectures.id
            JOIN courses ON lectures.course_id = courses.id
            WHERE attendance.student_id = %s
            ORDER BY attendance.marked_at DESC
            LIMIT 10
        """, (student_id,))
        history = cursor.fetchall()

        for h in history:
            if h.get('lecture_date'):
                h['lecture_date'] = h['lecture_date'].strftime('%d %b %Y')
            if h.get('marked_at'):
                h['marked_at'] = h['marked_at'].strftime('%I:%M %p')

        cursor.close()
        conn.close()

        return jsonify({"courses": result, "history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enroll/face', methods=['POST'])
def enroll_face():
    data = request.get_json()
    roll_no = data.get('roll_no')
    images = data.get('images')

    if not roll_no or not images or len(images) == 0:
        return jsonify({"error": "roll_no and images are required"}), 400

    try:
        save_path = f"dataset/{roll_no}"
        os.makedirs(save_path, exist_ok=True)

        existing_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
        start_count = len(existing_files)
        saved_count = 0

        for img_b64 in images:
            frame = decode_base64_image(img_b64)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            saved_count += 1
            cv2.imwrite(f"{save_path}/{start_count + saved_count}.jpg", face_img)

        if saved_count < 5:
            return jsonify({
                "error": f"Only {saved_count} clear face samples captured. Please try again with better lighting and look directly at the camera."
            }), 400

        retrain_model()

        return jsonify({
            "message": "Face enrolled and model updated successfully!",
            "samples_saved": saved_count
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/courses/all', methods=['GET'])
def get_all_courses():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM courses")
        courses = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"courses": courses}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/courses/available/<int:student_id>', methods=['GET'])
def get_available_courses(student_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT courses.id AS id, courses.course_name AS course_name, courses.course_code AS course_code
            FROM courses
            WHERE courses.id NOT IN (
                SELECT course_id FROM enrollments WHERE student_id = %s
            )
        """, (student_id,))
        courses = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"courses": courses}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enrollments/add', methods=['POST'])
def add_enrollment():
    data = request.get_json()
    student_id = data.get('student_id')
    course_id = data.get('course_id')

    if not all([student_id, course_id]):
        return jsonify({"error": "student_id and course_id are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO enrollments (student_id, course_id) VALUES (%s, %s)",
            (student_id, course_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Enrolled successfully!"}), 201
    except Exception as e:
        if "Duplicate entry" in str(e):
            return jsonify({"error": "You are already enrolled in this course"}), 400
        return jsonify({"error": str(e)}), 500

@app.route('/lectures/finalize', methods=['POST'])
def finalize_lecture():
    data = request.get_json()
    lecture_id = data.get('lecture_id')

    if not lecture_id:
        return jsonify({"error": "lecture_id is required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("UPDATE lectures SET status = 'completed' WHERE id = %s", (lecture_id,))

        cursor.execute("SELECT course_id FROM lectures WHERE id = %s", (lecture_id,))
        lecture_row = cursor.fetchone()
        if not lecture_row:
            cursor.close()
            conn.close()
            return jsonify({"error": "Lecture not found"}), 404
        course_id = lecture_row['course_id']

        cursor.execute("SELECT course_name, total_lectures FROM courses WHERE id = %s", (course_id,))
        course_row = cursor.fetchone()
        course_name = course_row['course_name']
        total_lectures = course_row['total_lectures'] or 45

        cursor.execute("""
            SELECT students.id AS id, students.name AS name
            FROM enrollments
            JOIN students ON enrollments.student_id = students.id
            WHERE enrollments.course_id = %s
        """, (course_id,))
        students = cursor.fetchall()

        flagged = []

        for student in students:
            cursor.execute("""
                SELECT COUNT(*) AS attended
                FROM attendance
                JOIN lectures ON attendance.lecture_id = lectures.id
                WHERE lectures.course_id = %s AND attendance.student_id = %s
            """, (course_id, student['id']))
            attended = cursor.fetchone()['attended']

            percentage = round((attended / total_lectures) * 100, 1) if total_lectures > 0 else 0

            if percentage < 65:
                status_type = 'not_eligible'
                message = (f"Your attendance in {course_name} is {percentage}%, below the 65% eligibility "
                           f"requirement. You are currently NOT ELIGIBLE for the final exam. Please contact your teacher.")
            elif percentage < 75:
                status_type = 'condonation'
                message = (f"Your attendance in {course_name} is {percentage}%, below the required 75%. "
                           f"You are in the CONDONATION zone. Please improve your attendance.")
            else:
                continue

            cursor.execute(
                "INSERT INTO notifications (student_id, course_id, message, status_type) VALUES (%s, %s, %s, %s)",
                (student['id'], course_id, message, status_type)
            )
            flagged.append({"name": student['name'], "percentage": percentage, "status_type": status_type})

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "message": "Lecture finalized and attendance checked.",
            "flagged_students": flagged
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/notifications/student/<int:student_id>', methods=['GET'])
def get_student_notifications(student_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT notifications.id AS id, notifications.message AS message,
                   notifications.status_type AS status_type, notifications.created_at AS created_at,
                   courses.course_name AS course_name
            FROM notifications
            JOIN courses ON notifications.course_id = courses.id
            WHERE notifications.student_id = %s
            ORDER BY notifications.created_at DESC
        """, (student_id,))
        notifications = cursor.fetchall()
        cursor.close()
        conn.close()

        for n in notifications:
            if n.get('created_at'):
                n['created_at'] = n['created_at'].strftime('%d %b %Y, %I:%M %p')

        return jsonify({"notifications": notifications}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/attendance/recognize', methods=['POST'])
def recognize_attendance():
    data = request.get_json()
    lecture_id = data.get('lecture_id')
    images = data.get('images')

    if not lecture_id or not images or len(images) == 0:
        return jsonify({"error": "lecture_id and images are required"}), 400

    try:
        # ---- Step 1: Liveness check across the captured frame sequence ----
        ear_sequence = get_ear_sequence(images)

        if len(ear_sequence) < 3:
            return jsonify({"error": "Could not track your face clearly. Please try again with better lighting."}), 400

        liveness_ok = detect_blink_in_sequence(ear_sequence)

        # ---- Step 2: Face recognition on a representative frame ----
        mid_index = len(images) // 2
        frame = decode_base64_image(images[mid_index])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            return jsonify({"error": "No face detected in image"}), 400

        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)

        if confidence >= CONFIDENCE_THRESHOLD:
            return jsonify({"error": "Face not recognized", "confidence": round(confidence, 2)}), 401

        if not liveness_ok:
            return jsonify({
                "error": "Liveness check failed. Please blink naturally during the scan.",
                "confidence": round(confidence, 2)
            }), 401

        roll_no = label_map.get(label)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students WHERE roll_no = %s", (roll_no,))
        student = cursor.fetchone()

        if not student:
            cursor.close()
            conn.close()
            return jsonify({"error": "Recognized roll_no not found in database"}), 404

        cursor.execute(
            "SELECT * FROM attendance WHERE lecture_id = %s AND student_id = %s",
            (lecture_id, student['id'])
        )
        existing = cursor.fetchone()

        if existing:
            cursor.close()
            conn.close()
            return jsonify({"message": f"Attendance already marked for {student['name']}"}), 200

        cursor.execute(
            "INSERT INTO attendance (lecture_id, student_id, status, liveness_verified) VALUES (%s, %s, %s, %s)",
            (lecture_id, student['id'], 'present', True)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "message": f"Attendance marked successfully for {student['name']}!",
            "student_name": student['name'],
            "roll_no": student['roll_no'],
            "confidence": round(confidence, 2)
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)