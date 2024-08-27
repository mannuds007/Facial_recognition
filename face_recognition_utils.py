import face_recognition
import os

# Path to the folder containing images
IMAGE_FOLDER = "D:/code/vv/robospeaker/facialRecognition/faces/"

def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(IMAGE_FOLDER):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            image = face_recognition.load_image_file(image_path)
            try:
                image_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(image_encoding)
                known_face_names.append(os.path.splitext(image_name)[0])
            except IndexError:
                print(f"Could not find a face in {image_name}. Skipping this file.")

    return known_face_encodings, known_face_names