import os
import pandas as pd
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer, process_words, Compose, RemovePunctuation, ToLowerCase
import torch
import torchaudio
import asyncio
import logging
import warnings
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions
import assemblyai as aai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import requests.exceptions
from audiomentations import Compose as AudioCompose, AddGaussianNoise, TimeStretch, PitchShift
import tempfile

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="inspect")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text normalization
text_transform = Compose([RemovePunctuation(), ToLowerCase()])

# Audio augmentation for noisy data
augment = AudioCompose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def run_whisper(waveform, sample_rate, model_name="openai/whisper-small"):
    """Run Whisper model on audio waveform."""
    logger.info(f"Running Whisper on audio")
    try:
        processor = WhisperProcessor.from_pretrained(model_name, use_fast=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return ""

def run_distil_whisper(waveform, sample_rate, model_name="distil-whisper/distil-large-v3"):
    """Run Distil-Whisper model on audio waveform."""
    logger.info(f"Running Distil-Whisper on audio")
    try:
        processor = WhisperProcessor.from_pretrained(model_name, use_fast=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        logger.error(f"Distil-Whisper error: {e}")
        return ""

def run_wav2vec2(waveform, sample_rate, model_name="facebook/wav2vec2-large-960h-lv60-self"):
    """Run Wav2Vec2 model on audio waveform."""
    logger.info(f"Running Wav2Vec2 on audio")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        logger.error(f"Wav2Vec2 error: {e}")
        return ""

async def run_deepgram(audio_data, sample_rate):
    """Run Deepgram transcription on audio data."""
    logger.info(f"Running Deepgram on audio")
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return "Deepgram API key not set"
    try:
        # Save audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            torchaudio.save(temp_file.name, torch.tensor(audio_data).unsqueeze(0), sample_rate)
            with open(temp_file.name, "rb") as audio:
                buffer_data = audio.read()
        
        client = DeepgramClient(api_key)
        options = PrerecordedOptions(model="nova-2", punctuate=True)
        response = client.listen.rest.v("1").transcribe_file(
            {"buffer": buffer_data, "mimetype": "audio/wav"}, options
        )
        transcription = response.results.channels[0].alternatives[0].transcript
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        return transcription
    except Exception as e:
        logger.error(f"Deepgram error: {e}")
        return ""

@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(10),
    retry=retry_if_exception_type((OSError, requests.exceptions.RequestException)),
    before_sleep=lambda retry_state: logger.info(f"Retrying AssemblyAI due to error: {retry_state.outcome.exception()}"),
)
def _assemblyai_transcribe(audio_data, sample_rate):
    """Synchronous helper function for AssemblyAI transcription."""
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        torchaudio.save(temp_file.name, torch.tensor(audio_data).unsqueeze(0), sample_rate)
        transcriber = aai.Transcriber()
        transcription = transcriber.transcribe(temp_file.name)
    os.unlink(temp_file.name)
    return transcription.text

async def run_assemblyai(audio_data, sample_rate):
    """Run AssemblyAI transcription on audio data with retries."""
    logger.info(f"Running AssemblyAI on audio")
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        return "AssemblyAI API key not set"
    try:
        transcription = await asyncio.to_thread(_assemblyai_transcribe, audio_data, sample_rate)
        return transcription
    except Exception as e:
        logger.error(f"AssemblyAI error: {e}")
        return ""

def run_deepspeech(audio_data, sample_rate):
    """Placeholder for DeepSpeech (not implemented)."""
    logger.info(f"DeepSpeech not supported on Python 3.11")
    return "DeepSpeech not supported"

async def benchmark_dataset(dataset_name="jaishah2808/speech-to-text-benchmark"):
    """Benchmark speech-to-text models on the Hugging Face dataset."""
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Initialize results
    results = []
    
    # Define models
    models = {
        "whisper": run_whisper,
        "distil_whisper": run_distil_whisper,
        "wav2vec2": run_wav2vec2,
        "deepgram": run_deepgram,
        "assemblyai": run_assemblyai,
        "deepspeech": run_deepspeech
    }
    
    for example in dataset['train']:  # Assuming 'train' split
        audio = example["audio"]
        true_transcription = example["text"]
        file_name = example.get("file_name", "unknown")
        
        # Clean audio
        clean_waveform = torch.tensor(audio["array"], dtype=torch.float32)
        clean_sample_rate = audio["sampling_rate"]
        
        # Noisy audio
        noisy_waveform = augment(clean_waveform.numpy(), sample_rate=clean_sample_rate)
        noisy_waveform = torch.tensor(noisy_waveform, dtype=torch.float32)
        
        for model_name, model_func in models.items():
            # Clean audio
            pred_clean = await model_func(clean_waveform, clean_sample_rate) if model_name in ["deepgram", "assemblyai"] else model_func(clean_waveform, clean_sample_rate)
            true_norm = text_transform(true_transcription)
            pred_norm_clean = text_transform(pred_clean) if pred_clean else ""
            error_rate_clean = wer(true_norm, pred_norm_clean) if pred_clean else float("inf")
            
            # Compute detailed WER measures for clean using process_words
            if pred_clean:
                output_clean = process_words(true_norm, pred_norm_clean)
                measures_clean = {
                    "insertions": output_clean.insertions,
                    "deletions": output_clean.deletions,
                    "substitutions": output_clean.substitutions
                }
            else:
                measures_clean = {"insertions": 0, "deletions": 0, "substitutions": 0}
            
            # Noisy audio
            pred_noisy = await model_func(noisy_waveform, clean_sample_rate) if model_name in ["deepgram", "assemblyai"] else model_func(noisy_waveform, clean_sample_rate)
            pred_norm_noisy = text_transform(pred_noisy) if pred_noisy else ""
            error_rate_noisy = wer(true_norm, pred_norm_noisy) if pred_noisy else float("inf")
            
            # Compute detailed WER measures for noisy using process_words
            if pred_noisy:
                output_noisy = process_words(true_norm, pred_norm_noisy)
                measures_noisy = {
                    "insertions": output_noisy.insertions,
                    "deletions": output_noisy.deletions,
                    "substitutions": output_noisy.substitutions
                }
            else:
                measures_noisy = {"insertions": 0, "deletions": 0, "substitutions": 0}
            
            # Store results
            results.append({
                "file_name": file_name,
                "model": model_name,
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
                "model": model_name,
                "category": "noisy",
                "true_transcription": true_transcription,
                "pred_transcription": pred_noisy,
                "wer": error_rate_noisy,
                "insertions": measures_noisy["insertions"],
                "deletions": measures_noisy["deletions"],
                "substitutions": measures_noisy["substitutions"]
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("advanced_benchmark_results.csv", index=False)
    logger.info(f"Benchmark results saved to advanced_benchmark_results.csv")
    
    # Compute average WER and error types per model and category
    avg_metrics = results_df.groupby(['model', 'category'])[['wer', 'insertions', 'deletions', 'substitutions']].mean().unstack()
    logger.info(f"Average metrics per model and category:\n{avg_metrics}")

if __name__ == "__main__":
    asyncio.run(benchmark_dataset())