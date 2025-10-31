import os
import tarfile
import pandas as pd
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

DATA_DIR = Path('../data')
RAW_DIR = DATA_DIR / 'raw'
# Delta segment directory name (note: '-delta-' in the name)
EXTRACT_DIR = RAW_DIR / 'cv-corpus-13.0-delta-2023-03-09'
ARCHIVE_PATH = RAW_DIR / 'en.tar.gz'

# 1. Extract the tar.gz if not already done
def extract_dataset():
    if EXTRACT_DIR.exists():
        print('Dataset already extracted.')
        return
    print('Extracting dataset. This will take a while...')
    with tarfile.open(ARCHIVE_PATH, 'r:gz') as tar:
        tar.extractall(RAW_DIR)
    print('Extraction complete.')

# 2. Filter samples for gender & create dataframe
def load_english_gendersplit(use_all_sources=True):
    """
    Load gender-labeled samples from Common Voice dataset.
    
    Args:
        use_all_sources: If True, load from validated, other, and invalidated TSV files
                        to get more female samples for balanced training.
    """
    clips_dir = EXTRACT_DIR / 'en/clips'
    
    # Load from validated.tsv (high quality)
    tsv_paths = [EXTRACT_DIR / 'en/validated.tsv']
    
    if use_all_sources:
        # Also include other.tsv and invalidated.tsv for more female samples
        tsv_paths.append(EXTRACT_DIR / 'en/other.tsv')
        tsv_paths.append(EXTRACT_DIR / 'en/invalidated.tsv')
        print("Loading data from validated, other, and invalidated TSV files...")
    else:
        print("Loading data from validated.tsv only...")
    
    all_dfs = []
    for tsv_path in tsv_paths:
        if tsv_path.exists():
            df_temp = pd.read_csv(tsv_path, sep='\t')
            df_temp = df_temp[df_temp['gender'].isin(['male', 'female'])]
            df_temp = df_temp[df_temp['path'].notnull()]
            df_temp['filepath'] = df_temp['path'].apply(lambda x: str(clips_dir / x))
            df_temp = df_temp[df_temp['filepath'].apply(lambda x: os.path.isfile(x))]
            all_dfs.append(df_temp[['filepath', 'gender']])
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=['filepath'])
    
    return df[['filepath', 'gender']]

# 3. Extract MFCC features with optional augmentation
def extract_mfcc(wav_path, sr=16000, n_mfcc=13, augment=False):
    """
    Extract MFCC features from audio file.
    
    Args:
        wav_path: Path to audio file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        augment: If True, apply augmentation (pitch shift, time stretch, noise)
    """
    try:
        y, sr = librosa.load(wav_path, sr=sr)
        
        # Apply augmentation for balancing dataset
        if augment:
            # Random pitch shift (simulates voice variations)
            pitch_shift = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            
            # Random time stretching (speed variation)
            time_stretch = np.random.uniform(0.95, 1.05)
            y = librosa.effects.time_stretch(y, rate=time_stretch)
            
            # Add slight noise for robustness
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, len(y))
            y = y + noise
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f'Error processing {wav_path}: {e}')
        return None

def main():
    extract_dataset()
    # Use all data sources to get more female samples
    df = load_english_gendersplit(use_all_sources=True)
    print(f'\nFound {len(df)} samples with gender labels.')
    male_count = len(df[df["gender"] == "male"])
    female_count = len(df[df["gender"] == "female"])
    print(f'Male: {male_count} ({100*male_count/len(df):.1f}%), Female: {female_count} ({100*female_count/len(df):.1f}%)')
    print(f'Balance ratio: {male_count/female_count:.2f}:1 (male:female)')
    
    X, y = [], []
    print('\nExtracting MFCC features from audio files...')
    print('Applying data augmentation to female samples for better balance...')
    print('This may take several minutes...\n')
    
    # Calculate augmentation to balance dataset better
    male_count = len(df[df["gender"] == "male"])
    female_count = len(df[df["gender"] == "female"])
    
    # More aggressive augmentation: aim for 2:1 ratio (male:female) instead of 1:1
    # This is faster while still significantly improving balance
    target_female = int(male_count / 2)  # Target 2:1 ratio
    augmentation_factor = max(2, int(target_female / female_count) - 1)
    
    print(f'Augmenting female samples {augmentation_factor}x to balance dataset (target: 2:1 ratio)...')
    
    total = len(df) + (female_count * augmentation_factor)
    processed = 0
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for idx, row in df.iterrows():
        if processed % 200 == 0:
            print(f'Progress: {processed}/{total} ({100*processed/total:.1f}%)')
        
        is_female = row['gender'] == 'female'
        
        # Always add original sample
        features = extract_mfcc(row['filepath'], augment=False)
        if features is not None:
            X.append(features)
            y.append(0 if row['gender'] == 'male' else 1)
            processed += 1
        
        # For female samples, add augmented versions
        if is_female:
            for aug_idx in range(augmentation_factor):
                if processed % 200 == 0:
                    print(f'Progress: {processed}/{total} ({100*processed/total:.1f}%)')
                
                features_aug = extract_mfcc(row['filepath'], augment=True)
                if features_aug is not None:
                    X.append(features_aug)
                    y.append(1)  # Female label
                    processed += 1
    
    print(f'\nSuccessfully processed {len(X)} samples.')
    np.savez(DATA_DIR / 'gender_features.npz', X=np.array(X), y=np.array(y))
    print(f'Saved features to: {DATA_DIR / "gender_features.npz"}')
    print(f'Feature shape: {np.array(X).shape}, Labels: {len(y)}')

if __name__ == '__main__':
    main()