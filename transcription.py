# Importation des librairies
import torchaudio
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC
import numpy as np
from datasets import load_dataset
from IPython.display import Audio
import librosa


# Charger un échantillon du dataset Common Voice en français
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="test")

# Choisir un échantillon audio (par exemple, le premier)
audio_sample = dataset[125]["audio"]

# Sauvegarder l'échantillon en tant que fichier .wav pour le test
file_path = "sample_audio.wav"
torchaudio.save(file_path, torch.tensor(audio_sample["array"]).unsqueeze(0), audio_sample["sampling_rate"])

print(f"Fichier audio sauvegardé : {file_path}")

signal, sr = librosa.load("sample_audio.wav")
Audio(signal, rate=sr)

# Fonction pour charger et préparer un fichier audio
def load_and_prepare_audio(file_path, target_sampling_rate=16000):
    # Charger le fichier audio
    waveform, original_sampling_rate = torchaudio.load(file_path)

    # S'assurer que l'audio est en format float32
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    # Rééchantillonnage si nécessaire
    if original_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
        waveform = resampler(waveform)
    
    # Retourner l'audio et la fréquence d'échantillonnage
    return waveform, target_sampling_rate

# Instanciation du modèle de transcription
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

# Charger et préparer l'audio
audio_input, sampling_rate = load_and_prepare_audio(file_path)

# Préparer l'entrée pour le modèle
inputs = processor(audio_input.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding="longest")

# Passer l'audio dans le modèle pour obtenir les logits
with torch.no_grad():
    logits = model(**inputs).logits

# Décodage pour obtenir la transcription
predicted_ids = logits.argmax(dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("Transcription : ", transcription)