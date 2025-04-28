from flask import Flask, request, Response, jsonify, render_template
import os
import base64
from datetime import datetime
from camera_opencv import Camera

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'photos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')


def gen(camera):
    "Video streaming generator function."
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = base64.b64decode(data['image'].split(',')[1])
        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)

        with open(path, 'wb') as f:
            f.write(image_data)

        return jsonify({"message": "Photo saved successfully!", "filename": filename}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
