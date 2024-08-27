import cv2
import numpy as np
import csv
from datetime import datetime
from face_recognition_utils import load_known_faces
import threading
import face_recognition

running = False
video_capture = None
face_recognition_thread = None
known_face_encodings, known_face_names = load_known_faces()

def face_recognition_process():
    global running, video_capture

    # Prepare CSV file for writing recognized faces
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    f = open(f"{current_date}.csv", "w+", newline="")
    lnwriter = csv.writer(f)

    video_capture = cv2.VideoCapture(0)

    while running:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            name = "Unknown"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name != "Unknown":
                lnwriter.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        cv2.imshow("Result", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()

def start_face_recognition():
    global running, face_recognition_thread
    if not running:
        running = True
        face_recognition_thread = threading.Thread(target=face_recognition_process)
        face_recognition_thread.start()
        return True
    return False

def stop_face_recognition():
    global running
    if running:
        running = False
        face_recognition_thread.join()
        return True
    return False

def reload_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces()
