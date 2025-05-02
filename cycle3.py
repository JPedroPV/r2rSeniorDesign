from flask import Flask, request, Response, jsonify, render_template, url_for
import os
import base64
import json
from datetime import datetime
from camera_opencv import Camera
from object_detection.opencvprotoPhoto import processPhotos

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'photos'
OUTPUT_FOLDER = 'static/output'  # Move output folder to static
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mainpage')
def mainpage():
    # Run the processPhotos function
    processPhotos(UPLOAD_FOLDER, OUTPUT_FOLDER)

    # Get the list of processed images (filtering out non-image files)
    processed_images = [
        url_for('static', filename=f'output/{img}')
        for img in os.listdir(OUTPUT_FOLDER)
        if img.endswith(".jpg") or img.endswith(".png")  # Only include image files
    ]

    # Load the results.json file to extract items and their counts
    results_file = os.path.join(OUTPUT_FOLDER, "results.json")
    item_counts = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
            for result in results:
                for obj in result["objects"]:
                    item_class = obj["class"]
                    if item_class in item_counts:
                        item_counts[item_class] += 1
                    else:
                        item_counts[item_class] = 1

    # Convert item_counts to a list of tuples for easier rendering in the template
    items = [(item, count) for item, count in item_counts.items()]

    return render_template('mainpage.html', images=processed_images, items=items)


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