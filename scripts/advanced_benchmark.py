import os
import random
import pandas as pd
from datasets import load_dataset
from jiwer import wer, Compose, RemovePunctuation, ToLowerCase, compute_measures
import torch
import torchaudio
import asyncio
import logging
import warnings
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from audiomentations import Compose as AudioCompose, AddGaussianNoise, TimeStretch, PitchShift

# Set seed
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

# Augmentation pipeline
class AudioAugment:
    def __init__(self):
        self.augment = AudioCompose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
        ])

    def __call__(self, waveform, sample_rate):
        return self.augment(waveform, sample_rate)

augment = AudioAugment()

class GraniteWrapper:
    def __init__(self, model_name="ibm-granite/granite-speech-3.3-8b"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True  # Required for Granite models
        )
        self.model.eval()
        self.sampling_rate = 16000  # Granite 3.3.1 expects 16kHz input
        
        # System prompt for the chat template
        self.system_prompt = (
            "Knowledge Cutoff Date: April 2024.\n"
            "Today's Date: June 14, 2025.\n"
            "You are Granite, developed by IBM. You are a helpful AI assistant"
        )

    def create_chat_prompt(self, transcription_request="can you transcribe the speech into a written format?"):
        """Create the chat template required by Granite Speech"""
        chat = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": f"<|audio|>{transcription_request}",
            }
        ]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def process_batch(self, waveforms, sample_rates):
        processed_results = []
        
        # Process each audio individually due to chat template requirements
        for wav, sr in zip(waveforms, sample_rates):
            try:
                # Ensure mono channel and correct sample rate
                if wav.dim() > 1:
                    wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)  # Add channel dimension
                
                if sr != self.sampling_rate:
                    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sampling_rate)
                
                # Ensure audio is at least 0.1s to avoid hallucination issues
                min_length = int(0.1 * self.sampling_rate)  # 0.1 seconds
                if wav.shape[1] < min_length:
                    # Pad with zeros if too short
                    pad_length = min_length - wav.shape[1]
                    wav = torch.nn.functional.pad(wav, (0, pad_length))
                
                # Create chat prompt
                text_prompt = self.create_chat_prompt()
                
                # Process with the model
                model_inputs = self.processor(
                    text_prompt,
                    wav,
                    device=DEVICE,
                    return_tensors="pt",
                ).to(self.model.device)
                
                with torch.no_grad():
                    model_outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=200,
                        num_beams=4,  # Use beam search as recommended
                        do_sample=False,
                        min_length=1,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        temperature=1.0,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Extract only the new tokens (response)
                num_input_tokens = model_inputs["input_ids"].shape[-1]
                new_tokens = model_outputs[0, num_input_tokens:]
                
                # Decode the response
                output_text = self.tokenizer.decode(
                    new_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                processed_results.append(output_text)
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                processed_results.append("")  # Return empty string on error
        
        return processed_results

granite_model = GraniteWrapper()

async def benchmark_dataset(
    dataset_name="jaishah2808/speech-to-text-benchmark",
    batch_size=4,  # Reduced batch size due to individual processing
    max_samples=None,
    run_idx=1
):
    set_seed(random.randint(0, 99999))
    
    try:
        dataset = load_dataset(dataset_name)
        samples = dataset['train']
        if max_samples:
            samples = samples.select(range(min(len(samples), max_samples)))
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    logger.info("Preloading audio data...")
    clean_waveforms, noisy_waveforms, sample_rates, texts = [], [], [], []

    for ex in tqdm(samples, desc="Preprocessing"):
        try:
            arr, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
            wav = torch.tensor(arr, dtype=torch.float32)
            
            # Ensure correct shape
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # Add channel dimension if missing
                
            clean_waveforms.append(wav)
            sample_rates.append(sr)
            texts.append(ex["text"])

            # Apply augmentations
            wav_np = wav.squeeze().cpu().numpy() if wav.dim() > 1 else wav.cpu().numpy()
            noisy_np = augment(wav_np, sr)
            noisy_wav = torch.tensor(noisy_np, dtype=torch.float32)
            if noisy_wav.dim() == 1:
                noisy_wav = noisy_wav.unsqueeze(0)
            noisy_waveforms.append(noisy_wav)
            
        except Exception as e:
            logger.error(f"Error preprocessing sample: {e}")
            continue

    results = []

    for mode, wf_list in [("clean", clean_waveforms), ("noisy", noisy_waveforms)]:
        logger.info(f"Processing {mode} audio...")
        preds = []
        
        for i in tqdm(range(0, len(wf_list), batch_size), desc=f"granite {mode}"):
            batch = wf_list[i:i + batch_size]
            sr_batch = sample_rates[i:i + batch_size]
            
            try:
                batch_preds = granite_model.process_batch(batch, sr_batch)
                preds.extend(batch_preds)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Add empty predictions for failed batch
                preds.extend([""] * len(batch))

        # Ensure we have predictions for all texts
        min_len = min(len(texts), len(preds))
        texts_subset = texts[:min_len]
        preds_subset = preds[:min_len]

        for text, pred in zip(texts_subset, preds_subset):
            try:
                ref = text_transform(text)
                hyp = text_transform(pred) if pred else ""
                measures = compute_measures(ref, hyp)
                results.append({
                    "model": "granite-speech-3.3",
                    "category": mode,
                    "true_transcription": text,
                    "transcription": pred,
                    "wer": measures['wer'],
                    "insertions": measures['insertions'],
                    "deletions": measures['deletions'],
                    "substitutions": measures['substitutions']
                })
            except Exception as e:
                logger.error(f"Error computing WER: {e}")
                results.append({
                    "model": "granite-speech-3.3",
                    "category": mode,
                    "true_transcription": text,
                    "transcription": pred,
                    "wer": 1.0,  # Maximum WER for failed cases
                    "insertions": 0,
                    "deletions": len(text.split()) if text else 0,
                    "substitutions": 0
                })

    df = pd.DataFrame(results)
    out_csv = f"granite_speech_benchmark_run{run_idx}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved to {out_csv}")
    
    # Print summary statistics
    try:
        summary = df.groupby(['model', 'category']).agg({
            'wer': ['mean', 'std', 'min', 'max'],
            'insertions': 'mean',
            'deletions': 'mean',
            'substitutions': 'mean'
        }).round(4)
        print("\n=== BENCHMARK RESULTS ===")
        print(summary)
        print(f"\nMean WER by category:")
        print(df.groupby(['model', 'category'])['wer'].mean().unstack().round(4))
    except Exception as e:
        logger.error(f"Error computing summary: {e}")

if __name__ == "__main__":
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    
    for run in range(1, 4):
        print(f"\n=== RUN {run} ===")
        try:
            asyncio.run(benchmark_dataset(batch_size=2, max_samples=100, run_idx=run))
        except Exception as e:
            logger.error(f"Error in run {run}: {e}")