import requests
from pathlib import Path

DATA_DIR = Path('../data')
RAW_DIR = DATA_DIR / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Common Voice dataset download URL
# NOTE: Direct download URLs may not work - download manually from:
# https://commonvoice.mozilla.org/en/datasets
#
# RECOMMENDED: Download "Common Voice Delta Segment 13.0" (~2.11 GB)
# Much smaller than full corpus (76 GB) but has same structure
#
# Save the downloaded file as 'en.tar.gz' in data/raw/ folder
#
# If direct URL works, uncomment one of these:

# Delta Segment 13.0 (~2.11 GB)
# CV_DOWNLOAD = 'https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-13.0-2023-03-09/cv-corpus-13.0-delta-2023-03-09-en.tar.gz'

# Or use a smaller Delta Segment:
# Delta Segment 17.0 (~1.6 GB)
# CV_DOWNLOAD = 'https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-17.0-2024-03-20/cv-corpus-17.0-delta-2024-03-20-en.tar.gz'

# Placeholder - set to None to skip download
CV_DOWNLOAD = None

def download_file(url, dest):
    if dest.exists():
        print(f"File {dest} already exists! Skipping download.")
        return
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)
    print(f"Downloaded {dest}")

def main():
    archive_path = RAW_DIR / 'en.tar.gz'
    
    if archive_path.exists():
        print(f"Dataset already exists at {archive_path}")
        print(f"Size: {archive_path.stat().st_size / (1024**3):.2f} GB")
        return
    
    if CV_DOWNLOAD is None:
        print("=" * 60)
        print("AUTOMATIC DOWNLOAD DISABLED")
        print("=" * 60)
        print("\nPlease download manually from:")
        print("https://commonvoice.mozilla.org/en/datasets")
        print("\nRECOMMENDED: 'Common Voice Delta Segment 13.0' (~2.11 GB)")
        print("or any Delta Segment version (much smaller than full corpus)")
        print(f"\nSave the file as: {archive_path}")
        print("=" * 60)
        return
    
    download_file(CV_DOWNLOAD, archive_path)

if __name__ == '__main__':
    main()