import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


video_capture = cv2.VideoCapture(0)

image_folder = "./faces/"


known_face_encodings = []
known_face_names = []


for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        try:
            image_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(image_encoding)
            known_face_names.append(os.path.splitext(image_name)[0])  # Use the file name (without extension) as the name
        except IndexError:
            print(f"Could not find a face in {image_name}. Skipping this file.")


now = datetime.now()
current_date = now.strftime("%d-%m-%Y")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)


face_locations = []
face_encodings = []

while True:
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

    # Display the resulting image
    cv2.imshow("Result", frame)

    # Quit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
