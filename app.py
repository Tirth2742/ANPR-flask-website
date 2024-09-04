from flask import Flask, Response, render_template, jsonify, send_file
import cv2
from video_processing import process_frame
from collections import deque
import time
import csv
import io

app = Flask(__name__)

camera = cv2.VideoCapture(0)
detected_plates = deque(maxlen=100)  # Store last 100 detected plates
last_detection = {}  # Dictionary to store the last detection time for each plate

def is_duplicate(plate_text, current_time):
    if plate_text in last_detection:
        time_diff = current_time - last_detection[plate_text]
        if time_diff < 60:  # 60 seconds = 1 minute
            return True
    last_detection[plate_text] = current_time
    return False

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_with_detections, plate_text = process_frame(frame)
            if plate_text:
                current_time = time.time()
                if not is_duplicate(plate_text, current_time):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    detected_plates.appendleft((plate_text, timestamp))
            ret, buffer = cv2.imencode('.jpg', frame_with_detections)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/get_plates')
def get_plates():
    return jsonify(list(detected_plates))

@app.route('/download_csv')
def download_csv():
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Number Plate', 'Date', 'Time'])
    for plate, timestamp in detected_plates:
        date, time = timestamp.split(' ')
        cw.writerow([plate, date, time])
    output = si.getvalue()
    si.close()
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=number_plates.csv"})

if __name__ == "__main__":
    app.run(debug=True)