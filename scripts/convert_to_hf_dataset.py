import os
import pandas as pd
from datasets import Dataset, Audio

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script's location
audio_dir = os.path.join(script_dir, "../audio_dataset/audio_files")
metadata_path = os.path.join(script_dir, "../audio_dataset/metadata.csv")
output_dir = os.path.join(script_dir, "../audio_dataset/hf_dataset")

# Print paths for debugging
print(f"Audio directory: {os.path.abspath(audio_dir)}")
print(f"Metadata path: {os.path.abspath(metadata_path)}")
print(f"Output directory: {os.path.abspath(output_dir)}")

# Load metadata
metadata = pd.read_csv(metadata_path)

# Add full file paths to the metadata
metadata['file_path'] = metadata['file_name'].apply(lambda x: os.path.join(audio_dir, x + '.wav'))

# Verify all audio files exist
missing_files = metadata[~metadata['file_path'].apply(os.path.exists)]
if not missing_files.empty:
    raise FileNotFoundError(f"Missing audio files: {missing_files['file_name'].tolist()}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(metadata)
dataset = dataset.cast_column("file_path", Audio())

# Save the dataset
os.makedirs(output_dir, exist_ok=True)
dataset.save_to_disk(output_dir)

print(f"Hugging Face dataset saved to {output_dir}")