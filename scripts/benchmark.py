# from datasets import load_from_disk
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from jiwer import wer, Compose, RemovePunctuation, ToLowerCase
# import torch
# import torchaudio
# import os
# import pandas as pd

# def run_whisper(audio_file, model_name="openai/whisper-small"):
#     # Print the file path for debugging
#     print(f"Attempting to load audio file: {audio_file}")
#     if not os.path.exists(audio_file):
#         raise FileNotFoundError(f"Audio file does not exist: {audio_file}")
    
#     # Load model and processor
#     processor = WhisperProcessor.from_pretrained(model_name)
#     model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
#     # Load audio
#     waveform, sample_rate = torchaudio.load(audio_file)
#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(sample_rate, 16000)
#         waveform = resampler(waveform)
    
#     # Process audio
#     input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
    
#     # Generate transcription
#     predicted_ids = model.generate(input_features)
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
#     return transcription

# def run_deepgram(audio_file):
#     return "Deepgram transcription not implemented"

# def run_assemblyai(audio_file):
#     return "AssemblyAI transcription not implemented"

# def run_google_speech(audio_file):
#     return "Google Speech-to-Text transcription not implemented"

# def run_azure_speech(audio_file):
#     return "Azure Speech-to-Text transcription not implemented"

# def run_groq(audio_file):
#     return "Groq transcription not implemented"

# def run_aws_transcribe(audio_file):
#     return "AWS Transcribe transcription not implemented"

# def benchmark_dataset(dataset_path="audio_dataset/hf_dataset"):
#     # Define the audio directory
#     audio_dir = os.path.join(os.path.dirname(__file__), "../audio_dataset/audio_files")
#     audio_dir = os.path.normpath(audio_dir)  # Normalize for Windows paths
    
#     # Load dataset
#     dataset = load_from_disk(dataset_path)
    
#     # Define text normalization: remove punctuation and convert to lowercase
#     transform = Compose([RemovePunctuation(), ToLowerCase()])
    
#     # Initialize results
#     results = []
    
#     # Run Whisper on each audio file
#     for example in dataset:
#         # Construct full path to audio file
#         audio_file = os.path.join(audio_dir, example["file_path"]["path"])
#         audio_file = os.path.normpath(audio_file)  # Normalize for Windows
#         true_transcription = example["transcription"]
        
#         # Run Whisper
#         pred_transcription = run_whisper(audio_file)
        
#         # Normalize texts
#         true_norm = transform(true_transcription)
#         pred_norm = transform(pred_transcription)
        
#         # Compute WER on normalized texts
#         error_rate = wer(true_norm, pred_norm)
        
#         results.append({
#             "file_name": os.path.basename(audio_file),
#             "true_transcription": true_transcription,
#             "pred_transcription": pred_transcription,
#             "wer": error_rate
#         })
    
#     # Save results
#     results_df = pd.DataFrame(results)
#     results_df.to_csv("benchmark_results.csv", index=False)
#     print(f"Benchmark results saved to benchmark_results.csv")
#     print(f"Average WER: {results_df['wer'].mean():.4f}")

# if __name__ == "__main__":
#     benchmark_dataset()

import os
import pandas as pd
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer, Compose, RemovePunctuation, ToLowerCase
import torch
import torchaudio
import asyncio
import logging
import warnings
from deepgram import DeepgramClient, PrerecordedOptions
import assemblyai as aai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import requests.exceptions

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="inspect")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
AUDIO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../audio_dataset/audio_files"))

# Text normalization
transform = Compose([RemovePunctuation(), ToLowerCase()])

def run_whisper(audio_file, model_name="openai/whisper-small"):
    """Run Whisper model on an audio file."""
    logger.info(f"Running Whisper on {audio_file}")
    try:
        processor = WhisperProcessor.from_pretrained(model_name, use_fast=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        waveform, sample_rate = torchaudio.load(audio_file)
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

def run_distil_whisper(audio_file, model_name="distil-whisper/distil-large-v3"):
    """Run Distil-Whisper model on an audio file."""
    logger.info(f"Running Distil-Whisper on {audio_file}")
    try:
        processor = WhisperProcessor.from_pretrained(model_name, use_fast=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        waveform, sample_rate = torchaudio.load(audio_file)
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

def run_wav2vec2(audio_file, model_name="facebook/wav2vec2-large-960h-lv60-self"):
    """Run Wav2Vec2 model on an audio file."""
    logger.info(f"Running Wav2Vec2 on {audio_file}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        waveform, sample_rate = torchaudio.load(audio_file)
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

async def run_deepgram(audio_file):
    """Run Deepgram transcription on an audio file."""
    logger.info(f"Running Deepgram on {audio_file}")
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return "Deepgram API key not set"
    try:
        client = DeepgramClient(api_key)
        with open(audio_file, "rb") as audio:
            buffer_data = audio.read()
        options = PrerecordedOptions(model="nova-2", punctuate=True)
        response = client.listen.rest.v("1").transcribe_file(
            {"buffer": buffer_data, "mimetype": "audio/wav"}, options
        )
        transcription = response.results.channels[0].alternatives[0].transcript
        return transcription
    except Exception as e:
        logger.error(f"Deepgram error: {e}")
        return ""

@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(10),
    retry=retry_if_exception_type((OSError, requests.exceptions.RequestException)),
    before_sleep=lambda retry_state: logger.info(f"Retrying AssemblyAI for {retry_state.fn.__code__.co_varnames[0]} due to error: {retry_state.outcome.exception()}"),
)
def _assemblyai_transcribe(audio_file):
    """Synchronous helper function for AssemblyAI transcription."""
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    return transcription.text

async def run_assemblyai(audio_file):
    """Run AssemblyAI transcription on an audio file with retries."""
    logger.info(f"Running AssemblyAI on {audio_file}")
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        return "AssemblyAI API key not set"
    try:
        transcription = await asyncio.to_thread(_assemblyai_transcribe, audio_file)
        return transcription
    except Exception as e:
        logger.error(f"AssemblyAI error: {e}")
        return ""

def run_deepspeech(audio_file):
    """Placeholder for DeepSpeech (not implemented)."""
    logger.info(f"DeepSpeech not supported on Python 3.11: {audio_file}")
    return "DeepSpeech not supported"

async def benchmark_dataset(dataset_path="audio_dataset/hf_dataset"):
    """Benchmark speech-to-text models on the dataset."""
    dataset = load_from_disk(dataset_path)
    results = []
    
    for example in dataset:
        audio_file = example["file_path"]["path"]
        audio_file = os.path.normpath(os.path.join(AUDIO_DIR, os.path.basename(audio_file)))
        true_transcription = example["transcription"]
        
        # Run each model
        transcriptions = {}
        transcriptions["whisper"] = run_whisper(audio_file)
        transcriptions["distil_whisper"] = run_distil_whisper(audio_file)
        transcriptions["wav2vec2"] = run_wav2vec2(audio_file)
        transcriptions["deepgram"] = await run_deepgram(audio_file)
        transcriptions["assemblyai"] = await run_assemblyai(audio_file)
        transcriptions["deepspeech"] = run_deepspeech(audio_file)
        
        # Compute WER for each model
        for model_name, pred_transcription in transcriptions.items():
            true_norm = transform(true_transcription)
            pred_norm = transform(pred_transcription) if pred_transcription else ""
            error_rate = wer(true_norm, pred_norm) if pred_transcription else float("inf")
            
            results.append({
                "file_name": os.path.basename(audio_file),
                "model": model_name,
                "true_transcription": true_transcription,
                "pred_transcription": pred_transcription,
                "wer": error_rate
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("benchmark_results.csv", index=False)
    logger.info(f"Benchmark results saved to benchmark_results.csv")
    logger.info(f"Average WER per model:\n{results_df.groupby('model')['wer'].mean().to_dict()}")

if __name__ == "__main__":
    asyncio.run(benchmark_dataset())