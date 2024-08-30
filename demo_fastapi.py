import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoTokenizer, BertModel, AutoProcessor, Wav2Vec2ForCTC
from torch.optim import Adam
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import io

app = FastAPI()

# Charger et préparer l'audio
def load_and_prepare_audio(file_path, target_sampling_rate=16000):
    waveform, original_sampling_rate = torchaudio.load(file_path)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    if original_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
        waveform = resampler(waveform)
    return waveform, target_sampling_rate

# Instanciation du modèle de transcription
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

# Modèle BERT personnalisé
class CustomBert(nn.Module):
    def __init__(self, name_or_model_path="bert-base-uncased", n_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

# Charger le modèle BERT préentraîné pour la classification des sentiments
bert_model = CustomBert()
bert_model.load_state_dict(torch.load("my_bert_exam_kk.pth", map_location=torch.device('cpu')))

# Classe pour la réponse JSON
class TranscriptionResponse(BaseModel):
    transcription: str
    sentiment: str

# Fonction principale pour la transcription et l'analyse de sentiment
@app.post("/transcribe_and_classify/", response_model=TranscriptionResponse)
async def process_audio_and_classify(file: UploadFile = File(...)):
    # Charger et préparer l'audio
    file_contents = await file.read()
    audio_input, sampling_rate = load_and_prepare_audio(io.BytesIO(file_contents))
    
    # Préparer l'entrée pour le modèle de transcription
    inputs = processor(audio_input.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
    
    # Obtenir les logits du modèle de transcription
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Décodage pour obtenir la transcription
    predicted_ids = logits.argmax(dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Classification du texte transcrit
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(transcription, padding="max_length", max_length=250, truncation=True, return_tensors="pt")
    output = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    _, pred = torch.max(output, dim=1)
    
    labels = {0: "negative", 1: "positive"}
    sentiment = labels[pred.item()]
    
    return TranscriptionResponse(transcription=transcription, sentiment=sentiment)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4446)
