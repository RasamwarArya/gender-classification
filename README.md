# Gender Classification from Audio

A machine learning project that classifies gender from audio recordings using MFCC features and a Random Forest classifier.

## Features

- **96.13% Accuracy** with balanced dataset
- **Web Interface** - Beautiful, user-friendly UI
- **Command Line Tools** - Simple CLI for predictions
- **Data Augmentation** - Automatic balancing for better female detection
- **Production Ready** - Fully trained model

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Web Interface

```bash
cd src
python app.py
```

Then open: **http://localhost:5000**

### Command Line Usage

```bash
cd src
python predict_gender.py <path_to_audio_file>
```

## Project Structure

```
src/
├── app.py                    # Web interface (Flask)
├── templates/
│   └── index.html           # Web UI
├── prepare_data.py          # Data preparation with augmentation
├── train_gender_classifier.py # Model training
├── predict_gender.py        # CLI prediction tool
└── retrain_with_augmentation.py # Retrain with augmentation

data/
├── models/                  # Trained models (gitignored - regenerated)
├── raw/                     # Dataset (gitignored - too large)
└── *.npz                    # Features (gitignored - regenerated)
```

## Model Performance

- **Overall Accuracy**: 96.13%
- **Female Recall**: 94.0% ✅
- **Male Recall**: 98.0% ✅
- **Female Precision**: 98.0% ✅

## Setup Instructions

1. Download dataset manually (see `src/download_common_voice.py` for instructions)
2. Run `python prepare_data.py` to process data
3. Run `python train_gender_classifier.py` to train model
4. Use `python app.py` to start web interface

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## License

Uses Common Voice dataset (CC-0 license)

