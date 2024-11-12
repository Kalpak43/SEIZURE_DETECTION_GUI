import os
import time
import mne
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response
from plot_edf import plot_edf_with_mne
from preprocess import segment_eeg  # Import segmentation and preprocessing functions

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('my_model.h5')

# Temporary directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# Route to upload the file
@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and file.filename.endswith('.edf'):
        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Return the file path so we can process it later
        return jsonify({"file_path": file_path})

    return jsonify({"error": "Invalid file type. Only EDF files are allowed."})




# # Route to process and predict from the uploaded file
# @app.route('/process_and_predict', methods=['POST'])
# def process_and_predict():
#     data = request.get_json()
#     file_path = data.get('file_path')  # Get the uploaded file path

#     if not file_path or not os.path.exists(file_path):
#         return jsonify({"error": "Invalid or missing file path."})

#     return Response(segment_eeg(file_path), content_type='text/event-stream')


# Route to stream predictions (GET)
@app.route('/stream_predictions', methods=['GET'])
def stream_predictions():
    # Get the file path from the query string
    file_path = request.args.get('file_path')  # e.g., ?file_path=path_to_your_uploaded_file.edf

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid or missing file path."}), 400

    return Response(segment_eeg(file_path), content_type='text/event-stream')


# Route to plot EDF file
@app.route('/plot_edf', methods=['POST'])
def plot_edf():
    data = request.get_json()
    file_path = data.get('file_path')
    
    print(file_path)

    if not file_path:
        return jsonify({"error": "File path is required"}), 400

    # Generate the plot
    img_base64 = plot_edf_with_mne(file_path)

    if img_base64 is None:
        return jsonify({"error": "Could not generate plot"}), 500

    return jsonify({"image": img_base64}), 200


if __name__ == '__main__':
    app.run(debug=True)

