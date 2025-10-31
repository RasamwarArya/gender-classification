from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
import joblib
from pathlib import Path
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load model
DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_PATH = DATA_DIR / 'models' / 'gender_classifier.joblib'

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        print(f'Error processing audio: {e}')
        return None

def predict_gender_from_file(audio_path):
    """Predict gender from audio file"""
    if model is None:
        return None, "Model not loaded"
    
    features = extract_mfcc(audio_path)
    if features is None:
        return None, "Failed to extract features"
    
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        gender = "Male" if prediction == 0 else "Female"
        confidence = probability[prediction] * 100
        
        return {
            'gender': gender,
            'confidence': round(confidence, 2),
            'probabilities': {
                'male': round(probability[0] * 100, 2),
                'female': round(probability[1] * 100, 2)
            }
        }, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict gender
            result, error = predict_gender_from_file(filepath)
            
            if error:
                return jsonify({'error': error}), 500
            
            # Clean up temporary file
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

