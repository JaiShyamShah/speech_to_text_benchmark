import os
import pandas as pd
from datasets import load_dataset
import torch
import torchaudio
import logging
import warnings
import numpy as np
import tempfile
import nemo.collections.asr as nemo_asr
from jiwer import wer, process_words, Compose, RemovePunctuation, ToLowerCase
from audiomentations import Compose as AudioCompose, AddGaussianNoise, TimeStretch, PitchShift
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="inspect")

# Text normalization
text_transform = Compose([RemovePunctuation(), ToLowerCase()])

# Audio augmentation for noisy data
augment = AudioCompose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# Global variable to store the loaded model
_asr_model = None

def load_model_with_retry(max_retries: int = 3, retry_delay: int = 30) -> Optional[object]:
    """Load the Parakeet model with retry logic and better error handling."""
    global _asr_model
    
    if _asr_model is not None:
        return _asr_model
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load Parakeet model (attempt {attempt + 1}/{max_retries})")
            
            # Load the model (NeMo will handle caching automatically)
            _asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
            logger.info("Model loaded successfully")
            return _asr_model
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All attempts to load the model failed")
                return None
    
    return None

def run_parakeet(waveform, sample_rate):
    """Run Parakeet-TDT-0.6B-v2 model on audio waveform."""
    logger.info("Running Parakeet on audio")
    
    # Load model if not already loaded
    asr_model = load_model_with_retry()
    if asr_model is None:
        logger.error("Failed to load ASR model")
        return ""
    
    try:
        # Ensure waveform is the right shape and type
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Ensure waveform is in the right format for saving
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            torchaudio.save(temp_file.name, waveform, sample_rate)
            
            # Transcribe
            try:
                output = asr_model.transcribe([temp_file.name])
                if output and len(output) > 0:
                    transcription = output[0] if isinstance(output[0], str) else output[0].text
                else:
                    transcription = ""
            except Exception as transcribe_error:
                logger.error(f"Transcription error: {transcribe_error}")
                transcription = ""
        
        # Clean up temporary file
        try:
            os.unlink(temp_file.name)
        except:
            pass  # Ignore cleanup errors
        
        return transcription
        
    except Exception as e:
        logger.error(f"Parakeet processing error: {e}")
        return ""

def benchmark_dataset(dataset_name="jaishah2808/speech-to-text-benchmark"):
    """Benchmark Parakeet-TDT-0.6B-v2 on the Hugging Face dataset."""
    
    try:
        # Load dataset from Hugging Face
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        logger.info(f"Dataset loaded successfully")
        
        # Get the appropriate split
        if 'train' in dataset:
            data_split = dataset['train']
        elif 'test' in dataset:
            data_split = dataset['test']
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            data_split = dataset[split_name]
            logger.info(f"Using split: {split_name}")
        
        logger.info(f"Processing full dataset with {len(data_split)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Pre-load the model before processing
    logger.info("Pre-loading ASR model...")
    if load_model_with_retry() is None:
        logger.error("Cannot proceed without ASR model")
        return
    
    # Initialize results
    results = []
    
    for i, example in enumerate(data_split):
        logger.info(f"Processing sample {i+1}/{len(data_split)}")
        
        try:
            audio = example["audio"]
            true_transcription = example["text"]
            file_name = example.get("file_name", f"sample_{i}")
            
            # Clean audio
            clean_waveform = torch.tensor(audio["array"], dtype=torch.float32)
            clean_sample_rate = audio["sampling_rate"]
            
            # Create noisy audio
            try:
                noisy_array = augment(clean_waveform.numpy(), sample_rate=clean_sample_rate)
                noisy_waveform = torch.tensor(noisy_array, dtype=torch.float32)
            except Exception as aug_error:
                logger.warning(f"Audio augmentation failed: {aug_error}, using original audio")
                noisy_waveform = clean_waveform.clone()
            
            # Run Parakeet on clean audio
            pred_clean = run_parakeet(clean_waveform, clean_sample_rate)
            true_norm = text_transform(true_transcription)
            pred_norm_clean = text_transform(pred_clean) if pred_clean else ""
            
            # Calculate WER for clean audio
            if pred_clean and true_norm:
                error_rate_clean = wer(true_norm, pred_norm_clean)
                output_clean = process_words(true_norm, pred_norm_clean)
                measures_clean = {
                    "insertions": output_clean.insertions,
                    "deletions": output_clean.deletions,
                    "substitutions": output_clean.substitutions
                }
            else:
                error_rate_clean = float("inf")
                measures_clean = {"insertions": 0, "deletions": 0, "substitutions": 0}
            
            # Run Parakeet on noisy audio
            pred_noisy = run_parakeet(noisy_waveform, clean_sample_rate)
            pred_norm_noisy = text_transform(pred_noisy) if pred_noisy else ""
            
            # Calculate WER for noisy audio
            if pred_noisy and true_norm:
                error_rate_noisy = wer(true_norm, pred_norm_noisy)
                output_noisy = process_words(true_norm, pred_norm_noisy)
                measures_noisy = {
                    "insertions": output_noisy.insertions,
                    "deletions": output_noisy.deletions,
                    "substitutions": output_noisy.substitutions
                }
            else:
                error_rate_noisy = float("inf")
                measures_noisy = {"insertions": 0, "deletions": 0, "substitutions": 0}
            
            # Store results
            results.append({
                "file_name": file_name,
                "model": "parakeet",
                "category": "clean",
                "true_transcription": true_transcription,
                "pred_transcription": pred_clean,
                "wer": error_rate_clean,
                "insertions": measures_clean["insertions"],
                "deletions": measures_clean["deletions"],
                "substitutions": measures_clean["substitutions"]
            })
            
            results.append({
                "file_name": file_name,
                "model": "parakeet",
                "category": "noisy",
                "true_transcription": true_transcription,
                "pred_transcription": pred_noisy,
                "wer": error_rate_noisy,
                "insertions": measures_noisy["insertions"],
                "deletions": measures_noisy["deletions"],
                "substitutions": measures_noisy["substitutions"]
            })
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} samples")
                
        except Exception as sample_error:
            logger.error(f"Error processing sample {i}: {sample_error}")
            continue
    
    if not results:
        logger.error("No results to save")
        return
    
    # Save results
    try:
        results_df = pd.DataFrame(results)
        output_file = "parakeet_benchmark_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Benchmark results saved to {output_file}")
        
        # Compute and display average metrics
        avg_metrics = results_df.groupby(['category'])[['wer', 'insertions', 'deletions', 'substitutions']].mean()
        logger.info(f"Average metrics per category:\n{avg_metrics}")
        
        # Display summary statistics
        summary_stats = results_df.groupby(['category'])['wer'].agg(['mean', 'std', 'min', 'max']).round(4)
        logger.info(f"WER summary statistics:\n{summary_stats}")
        
    except Exception as save_error:
        logger.error(f"Error saving results: {save_error}")

if __name__ == "__main__":
    benchmark_dataset()