from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn
import mysql.connector
import threading
import time
import numpy as np
import cv2
import requests
from ultralytics import YOLO
import acconeer.exptool as et
from acconeer.exptool import a121
from acconeer.exptool.a121.algo.presence import Detector, DetectorConfig

app = FastAPI()

# Connect to MariaDB/MySQL
db = mysql.connector.connect(
    host="localhost",
    user="soo",
    password="a",
    database="radar_detection"
)
cursor = db.cursor()

# Shared data
latest_data = {"distance_m": None}
radar_lock = threading.Lock()
radar_distance = None
latest_frame = None
frame_lock = threading.Lock()

# Radar thread
def radar_thread():
    global radar_distance
    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)
    client = a121.Client.open(**a121.get_client_args(args))

    detector_config = DetectorConfig(start_m=0.3, end_m=4.0)
    detector = Detector(client=client, sensor_id=1, detector_config=detector_config)
    detector.start()

    try:
        while True:
            result = detector.get_next()
            with radar_lock:
                if result.presence_detected:
                    radar_distance = result.presence_distance
                else:
                    radar_distance = None
    finally:
        detector.stop()
        client.close()

# YOLO + camera + POST to self
def processing_thread():
    global latest_frame
    threading.Thread(target=radar_thread, daemon=True).start()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        print("Error: Could not open /dev/video0")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, save=False, conf=0.3, verbose=False)
        person_detected = False

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = f"Person {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        with radar_lock:
            current_distance = radar_distance

        if person_detected and current_distance is not None:
            text = f"Person detected at {current_distance:.2f} m"
            distance_data = current_distance
        else:
            text = "No person detected"
            distance_data = None

        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        with frame_lock:
            latest_frame = frame.copy()

        try:
            requests.post("http://127.0.0.1:8001/data",
                          json={"distance_m": distance_data}, timeout=0.3)
        except:
            pass

        time.sleep(0.03)

from fastapi.responses import StreamingResponse

def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)  # ~30 fps

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")



# FastAPI endpoints
@app.post("/data")
async def receive_data(request: Request):
    body = await request.json()
    distance = body.get("distance_m")
    cursor.execute("INSERT INTO detection_log (distance_m) VALUES (%s)", (distance,))
    db.commit()
    latest_data["distance_m"] = distance
    print(f"Stored in DB: distance_m={distance}")
    return JSONResponse(content={"status":"ok"})

@app.get("/status")
async def get_status():
    return JSONResponse(content=latest_data)

@app.get("/camera")
async def get_camera_snapshot():
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return Response(content="Camera not ready", media_type="text/plain", status_code=503)
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Radar Dashboard</title>
<style>
body { font-family: Arial; text-align: center; background: #f8f9fa; padding: 40px;}
.line-container { position: relative; width: 80%; height: 4px; background: #333; margin: 20px auto;}
.dot { position: absolute; top:-8px; width:16px; height:16px; background: red; border-radius:50%; transition: left 0.5s;}
.scale { display: flex; justify-content: space-between; width:80%; margin: 10px auto 30px auto; font-size:14px; color:#555;}
#status { margin-top: 20px; font-size:22px;}
</style>
</head>
<body>
<h1>Real-time Radar Dashboard</h1>
<div class="scale"><span>0 m</span><span>1 m</span><span>2 m</span><span>3 m</span><span>4 m</span></div>
<div class="line-container" id="line"><div class="dot" id="dot" style="display:none;"></div></div>
<div id="status">Loading...</div>
<h2>Live Camera Feed</h2>
<img id="camera" src="/camera" style="width:640px;">
<script>
const maxDistance = 4.0;
const line = document.getElementById("line");
const dot = document.getElementById("dot");
const statusText = document.getElementById("status");
async function update() {
    try {
        const res = await fetch("/status");
        const data = await res.json();
        const distance = data.distance_m;
        if (distance !== null) {
            statusText.textContent = `Person detected at ${distance.toFixed(2)} m`;
            let lineWidth = line.clientWidth;
            let pos = Math.min(distance/maxDistance, 1.0) * lineWidth;
            dot.style.left = (pos - dot.clientWidth/2) + "px";
            dot.style.display = "block";
        } else {
            statusText.textContent = "No person detected";
            dot.style.display = "none";
        }
    } catch {
        statusText.textContent = "Server error";
        dot.style.display = "none";
    }
    document.getElementById("camera").src = "/camera?ts=" + new Date().getTime();
}
setInterval(update, 200);
update();
</script>
</body>
</html>
""")

# Run everything
if __name__ == "__main__":
    threading.Thread(target=processing_thread, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8001)
