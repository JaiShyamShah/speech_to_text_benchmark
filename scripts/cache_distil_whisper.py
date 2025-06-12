from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_name = "distil-whisper/distil-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
print("Model cached successfully")