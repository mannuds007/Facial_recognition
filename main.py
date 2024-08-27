from fastapi import FastAPI
from face_recognition_service import (
    start_face_recognition,
    stop_face_recognition,
    reload_known_faces,
    known_face_names
)

app = FastAPI()

@app.post("/start")
async def start_recognition():
    if start_face_recognition():
        return {"status": "Face recognition started."}
    else:
        return {"status": "Face recognition is already running."}

@app.post("/stop")
async def stop_recognition():
    if stop_face_recognition():
        return {"status": "Face recognition stopped."}
    else:
        return {"status": "Face recognition is not running."}

@app.get("/faces")
async def get_faces():
    return {"known_faces": known_face_names}

@app.post("/reload_faces")
async def reload_faces():
    reload_known_faces()
    return {"status": "Faces reloaded."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
