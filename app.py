from flask import Flask, request, render_template, jsonify
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from audio import processing

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GRAPH_FOLDER'] = 'graphs'
# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['GRAPH_FOLDER']):
    os.makedirs(app.config['GRAPH_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # Save the uploaded file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded file using librosa (or other audio processing libraries)
        audio, sr = librosa.load(file_path)
        # Perform your audio processing here
        loudness = processing.process_audio(audio)
        spectrogram = processing.spectrogram(audio)
        stft = processing.stft(audio)
        melspec = processing.getMelSpec(audio)
        chroma = processing.getChroma(audio)

        # Optionally, you can return processing results
        # For example:
        # loudness = your_processing_function(audio)
        # return jsonify({'loudness': loudness})

        return render_template('results.html',loudness = loudness, spectrogram = spectrogram, stft = stft, melspec = melspec, chroma = chroma)

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
