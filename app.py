from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort, jsonify
import os
import face_recognition
import numpy as np
import uuid
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
executor = ThreadPoolExecutor()

known_face_encodings = []
known_face_names = []
jobs = {}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_image(image_path, size=(500, 500)):
    image = Image.open(image_path).convert('RGB')
    image.thumbnail(size, Image.LANCZOS)
    return np.array(image)

def create_person_folder(person_name):
    person_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'known', person_name)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder

def load_and_encode_images():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    known_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'known')
    for person_name in os.listdir(known_dir):
        person_folder = os.path.join(known_dir, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, filename)
                    image = preprocess_image(img_path)
                    face_locations = face_recognition.face_locations(image, model='hog')
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                    else:
                        logging.debug(f"No faces found in the image: {filename}")

def compare_faces_batch(test_image_path):
    image = face_recognition.load_image_file(test_image_path)
    face_locations = face_recognition.face_locations(image, model='hog')
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    
    results = {}
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < 0.4:  # Adjust this threshold as needed
            name = known_face_names[best_match_index]
            confidence = 1 - face_distances[best_match_index]
            
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
            
            text_bbox = draw.textbbox((left, bottom), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)
            
            # Update results, keeping only the highest confidence for each person
            if name not in results or confidence > results[name]['confidence']:
                results[name] = {
                    "name": name,
                    "location": (left, top, right, bottom),
                    "confidence": confidence
                }
            
            logging.debug(f"Match found: {name} with confidence {confidence:.2f}")
        else:
            logging.debug(f"No match found for face at {(left, top, right, bottom)}")
    
    result_image_path = os.path.join('results', f"result_{os.path.basename(test_image_path)}")
    full_result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_path)
    pil_image.save(full_result_image_path)
    
    logging.debug(f"Result image saved to: {full_result_image_path}")
    return list(results.values()), result_image_path

def process_image_task(test_image_path):
    time.sleep(2)  # Simulate long processing time
    results, result_image_path = compare_faces_batch(test_image_path)
    return results, result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_known', methods=['GET', 'POST'])
def upload_known():
    if request.method == 'POST':
        person_name = request.form.get('person_name')
        if not person_name:
            return "Person name is required", 400

        if 'known_images' not in request.files:
            return "No file part", 400

        known_images = request.files.getlist('known_images')
        if not known_images or known_images[0].filename == '':
            return "No selected file", 400

        person_folder = create_person_folder(person_name)
        for known_image in known_images:
            if known_image and allowed_file(known_image.filename):
                filename = secure_filename(f"{uuid.uuid4()}_{known_image.filename}")
                filepath = os.path.join(person_folder, filename)
                known_image.save(filepath)

        executor.submit(load_and_encode_images)
        return redirect(url_for('index'))

    return render_template('upload_known.html')

@app.route('/upload_test', methods=['GET', 'POST'])
def upload_test():
    if request.method == 'POST':
        if 'test_image' not in request.files:
            return jsonify({"error": "No file part"}), 400

        test_image = request.files['test_image']
        if test_image.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if test_image and allowed_file(test_image.filename):
            filename = secure_filename(f"{uuid.uuid4()}_{test_image.filename}")
            test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test', filename)
            test_image.save(test_image_path)

            job_id = str(uuid.uuid4())
            job = executor.submit(process_image_task, test_image_path)
            jobs[job_id] = job
            return jsonify({"job_id": job_id}), 202

    return render_template('upload_test.html')

@app.route('/job_status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    
    if job.done():
        results, result_image_path = job.result()
        del jobs[job_id]  # Remove the job from the dictionary
        return jsonify({
            "status": "complete",
            "results": results,
            "result_image_path": result_image_path
        })
    else:
        return jsonify({"status": "processing"})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'known'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'test'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)
    threading.Thread(target=load_and_encode_images).start()
    app.run(debug=True)
