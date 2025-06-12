import os
import random
import pandas as pd
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer, Compose, RemovePunctuation, ToLowerCase
import torch
import torchaudio
import asyncio
import logging
import warnings
import numpy as np
from audiomentations import Compose as AudioCompose, AddGaussianNoise, TimeStretch, PitchShift
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

text_transform = Compose([RemovePunctuation(), ToLowerCase()])

class AudioAugmentGPU:
    def __init__(self):
        self.noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        self.stretch = TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
        self.pitch = PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
    def __call__(self, waveform, sample_rate):
        waveform = self.noise(waveform, sample_rate)
        if np.random.rand() < 0.5:
            waveform = self.stretch(waveform, sample_rate)
        if np.random.rand() < 0.5:
            waveform = self.pitch(waveform, sample_rate)
        return waveform

augment = AudioAugmentGPU()

class WhisperWrapper:
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(DEVICE)
    def process_batch(self, waveforms, sample_rates, temperature=0.3):
        processed = []
        for wav, sr in zip(waveforms, sample_rates):
            wav = wav.to(DEVICE)
            if sr != 16000:
                wav = self.resampler(wav)
            processed.append(wav.squeeze().cpu().numpy())
        inputs = self.processor(
            processed, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True,
            return_attention_mask=True
        ).input_features.to(DEVICE)
        if inputs.shape[-1] < 3000:
            inputs = torch.nn.functional.pad(inputs, (0, 3000 - inputs.shape[-1]))
        with torch.no_grad():
            outputs = self.model.generate(inputs, temperature=temperature)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)

class Wav2Vec2Wrapper:
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(DEVICE)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(DEVICE)
    def process_batch(self, waveforms, sample_rates):
        processed = []
        for wav, sr in zip(waveforms, sample_rates):
            wav = wav.to(DEVICE)
            if sr != 16000:
                wav = self.resampler(wav)
            processed.append(wav.squeeze().cpu().numpy())
        inputs = self.processor(
            processed, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).input_values.to(DEVICE)
        with torch.no_grad():
            logits = self.model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)

whisper_model = WhisperWrapper()
distil_whisper_model = WhisperWrapper("distil-whisper/distil-large-v3")
wav2vec2_model = Wav2Vec2Wrapper()

async def benchmark_dataset(
    dataset_name="jaishah2808/speech-to-text-benchmark",
    batch_size=16,
    max_samples=None,
    temperature=0.3,
    run_idx=1
):
    set_seed(42)
    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train']
    dataset_size = len(train_dataset)
    logger.info(f"Train dataset size: {dataset_size}")
    if max_samples:
        n_samples = min(max_samples, dataset_size)
        samples = train_dataset.select(range(n_samples))
    else:
        samples = train_dataset
    logger.info("Preloading audio data...")
    clean_waveforms, noisy_waveforms, sample_rates, texts = [], [], [], []
    for example in tqdm(samples, desc="Preprocessing"):
        audio = example["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]
        if DEVICE == "cuda":
            waveform = waveform.cuda()
        clean_waveforms.append(waveform)
        noisy_waveforms.append(torch.tensor(augment(waveform.cpu().numpy(), sr)).to(DEVICE))
        sample_rates.append(sr)
        texts.append(example["text"])
    results = []
    model_funcs = {
        "whisper": lambda x, y: whisper_model.process_batch(x, y, temperature=temperature),
        "distil_whisper": lambda x, y: distil_whisper_model.process_batch(x, y, temperature=temperature),
        "wav2vec2": wav2vec2_model.process_batch,
    }
    for model_name, model_fn in model_funcs.items():
        logger.info(f"Processing {model_name} in batches...")
        clean_preds = []
        for i in tqdm(range(0, len(clean_waveforms), batch_size), desc=f"{model_name} clean"):
            batch = clean_waveforms[i:i+batch_size]
            sr_batch = sample_rates[i:i+batch_size]
            clean_preds.extend(model_fn(batch, sr_batch))
        noisy_preds = []
        for i in tqdm(range(0, len(noisy_waveforms), batch_size), desc=f"{model_name} noisy"):
            batch = noisy_waveforms[i:i+batch_size]
            sr_batch = sample_rates[i:i+batch_size]
            noisy_preds.extend(model_fn(batch, sr_batch))
        for i, (text, pred_clean, pred_noisy) in enumerate(zip(texts, clean_preds, noisy_preds)):
            results.append({
                "model": model_name,
                "category": "clean",
                "wer": wer(text_transform(text), text_transform(pred_clean)),
                "transcription": pred_clean
            })
            results.append({
                "model": model_name,
                "category": "noisy",
                "wer": wer(text_transform(text), text_transform(pred_noisy)),
                "transcription": pred_noisy
            })
    df = pd.DataFrame(results)
    out_csv = f"gpu_optimized_results_run{run_idx}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved to {out_csv}")
    print(df.groupby(['model', 'category']).wer.mean().unstack())

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    for run in range(1, 4):
        print(f"\n=== RUN {run} ===")
        asyncio.run(benchmark_dataset(batch_size=16, max_samples=100, temperature=0.3, run_idx=run))
