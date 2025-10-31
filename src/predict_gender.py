import numpy as np
from pathlib import Path
import librosa
import joblib
import sys

DATA_DIR = Path('../data')
MODEL_DIR = DATA_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'gender_classifier.joblib'

def extract_mfcc(wav_path, sr=16000, n_mfcc=13):
    """Extract MFCC features from audio file (same as prepare_data.py)"""
    try:
        y, sr = librosa.load(wav_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.reshape(1, -1)  # Reshape for single prediction
    except Exception as e:
        print(f'Error processing {wav_path}: {e}')
        return None

def predict_gender(audio_path, model_path=None):
    """Predict gender from audio file"""
    if model_path is None:
        model_path = MODEL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please run train_gender_classifier.py first to train the model."
        )
    
    # Load model
    model = joblib.load(model_path)
    
    # Extract features
    features = extract_mfcc(audio_path)
    if features is None:
        return None
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    gender = "Male" if prediction == 0 else "Female"
    confidence = probability[prediction] * 100
    
    return {
        'gender': gender,
        'confidence': confidence,
        'probabilities': {
            'male': probability[0] * 100,
            'female': probability[1] * 100
        }
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_gender.py <audio_file_path>")
        print("Example: python predict_gender.py ../data/test_audio.wav")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"Processing: {audio_path}")
    result = predict_gender(audio_path)
    
    if result:
        print(f"\n{'='*50}")
        print("Prediction Result")
        print(f"{'='*50}")
        print(f"Gender: {result['gender']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Probabilities:")
        print(f"  Male: {result['probabilities']['male']:.2f}%")
        print(f"  Female: {result['probabilities']['female']:.2f}%")
    else:
        print("Failed to process audio file.")

if __name__ == '__main__':
    main()

