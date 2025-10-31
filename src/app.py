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
# With Root Directory = 'src' in Render, __file__ is in src/, so we go up one level to repo root
DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_PATH = DATA_DIR / 'models' / 'gender_classifier.joblib'

# Alternative path if above doesn't work (absolute from repo root)
if not MODEL_PATH.exists():
    # Try from current working directory (should be repo root in Render)
    MODEL_PATH = Path('data') / 'models' / 'gender_classifier.joblib'

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        # Try from current working directory (repo root)
        alt_path = Path('data') / 'models' / 'gender_classifier.joblib'
        if alt_path.exists():
            MODEL_PATH = alt_path
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print(f"Also checked: {alt_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current dir: {os.listdir('.')}")
            model = None
except Exception as e:
    import traceback
    print(f"Error loading model: {e}")
    print(f"Traceback: {traceback.format_exc()}")
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
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not file.filename:
            return jsonify({'error': 'Invalid file'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        # Check if model is loaded
        if model is None:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        try:
            # Predict gender
            result, error = predict_gender_from_file(filepath)
            
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if error:
                return jsonify({'error': error}), 500
            
            if result is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            import traceback
            error_msg = f'Error processing file: {str(e)}'
            print(f'Prediction error: {traceback.format_exc()}')
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        import traceback
        error_msg = f'Server error: {str(e)}'
        print(f'Server error: {traceback.format_exc()}')
        return jsonify({'error': error_msg}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

