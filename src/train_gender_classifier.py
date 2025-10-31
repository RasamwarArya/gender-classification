import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATA_DIR = Path('../data')
MODEL_DIR = DATA_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / 'gender_features.npz'
MODEL_PATH = MODEL_DIR / 'gender_classifier.joblib'

def load_features():
    """Load preprocessed MFCC features and labels"""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_PATH}\n"
            "Please run prepare_data.py first to generate features."
        )
    
    data = np.load(FEATURES_PATH)
    X = data['X']
    y = data['y']
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"Class distribution: {np.bincount(y)} (0=male, 1=female)")
    
    return X, y

def train_model(X_train, y_train, model_type='random_forest', use_class_weights=True):
    """Train a gender classification model with improved parameters for female detection
    
    Args:
        use_class_weights: If True, use balanced class weights to handle imbalanced data
    """
    if model_type == 'random_forest':
        # Use class weights to balance the imbalanced dataset
        # 'balanced_subsample' works better for RandomForest - applies balancing at each bootstrap
        class_weight = 'balanced_subsample' if use_class_weights else None
        
        # Increased estimators and max_depth for better female detection
        model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100 for better performance
            max_depth=25,  # Increased from 20
            min_samples_split=5,  # Prevent overfitting
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight,  # Better balancing for female detection
            bootstrap=True
        )
    elif model_type == 'svm':
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining {model_type} model...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("Model Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))
    
    return accuracy, y_pred

def main():
    # Load features
    X, y = load_features()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model with balanced class weights for better female detection
    print("\nUsing balanced class weights to handle imbalanced dataset...")
    model = train_model(X_train, y_train, model_type='random_forest', use_class_weights=True)
    
    # Evaluate
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()

