#Face Recognition for attendance

This project is a face recognition API built using FastAPI and OpenCV. It uses a webcam to capture video, recognizes faces based on pre-loaded images from a specified folder, and logs recognized faces to a CSV file.

## Features

- Start and stop face recognition via API.
- Load and reload known faces from a folder.
- Log recognized faces with timestamps in a CSV file.

## Requirements

- Python 3.7+
- Required packages: FastAPI, Uvicorn, OpenCV, face_recognition

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the server**: `uvicorn main:app --reload`
3. **Use API endpoints** to control face recognition:
   - `POST /start` - Start recognition
   - `POST /stop` - Stop recognition
   - `GET /faces` - List known faces
   - `POST /reload_faces` - Reload faces
