import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer, AutoModelForMaskedLM
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import gradio as gr



class AlloCineDataset(Dataset):
  def __init__(self, csv_file, device, model_name_or_path="bert-base-uncased", max_length=2000):
    self.device = device
    self.df = pd.read_csv(csv_file)
    self.labels=self.df.polarity.unique()
    labels_dict=dict()
    for indx, l in enumerate(self.labels):
      labels_dict[l]=indx

    self.df["polarity"]=self.df["polarity"].map(labels_dict)
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    review = self.df.review[index]
    label_review = self.df.polarity[index]

    inputs = self.tokenizer(review, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
    labels = torch.tensor(label_review)

    return {
        "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
        "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
        "labels": labels.to(self.device)
    }



class CustomBert(nn.Module):
  def __init__(self, name_or_model_path="bert-base-uncased", n_classes=2):
    super(CustomBert, self).__init__()
    self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
    self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
    x = self.classifier(x.pooler_output)

    return x

  def save_checkpoint(self, path):
    torch.save(self.state_dict(), path)


def training_step(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    total_samples = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        optimizer.zero_grad()

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)  # Multiplier par le nombre d'échantillons dans le batch
        total_samples += labels.size(0)

    return total_loss / total_samples  # Diviser par le nombre total d'échantillons

def evaluation(model, data_loader, loss_fn):
    model.eval()
    correct_predictions = 0
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred = torch.max(output, dim=1)

            correct_predictions += torch.sum(pred == labels)
            total_loss += loss_fn(output, labels).item() * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples, (correct_predictions.double() / total_samples)*100



def main():
  print("Training ....")
  N_EPOCHS = 4
  LR = 2e-5
  BATCH_SIZE = 128

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_dataset = AlloCineDataset(csv_file="/kaggle/input/allocine-french-movie-reviews/train.csv", device=device, max_length=100)
  test_dataset = AlloCineDataset(csv_file="/kaggle/input/allocine-french-movie-reviews/test.csv", device=device, max_length=100)
  validation_dataset = AlloCineDataset(csv_file="/kaggle/input/allocine-french-movie-reviews/valid.csv", device=device, max_length=100)

  train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
  test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
  validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE)

  model = CustomBert()
  model=model.to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr =LR)

  for epoch in range(N_EPOCHS):
    loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
    loss_eval, accuracy_eval = evaluation(model, test_dataloader, loss_fn)
    loss_val, accuracy_val = evaluation(model, validation_dataloader, loss_fn)

    print(f"Epoch {epoch+1}/{N_EPOCHS}")
    print(f"Train Loss : {loss_train} | Test Loss : {loss_eval} | Validation Loss : {loss_val}" )
    print(f"Test Accuracy : {accuracy_eval} | Validation Accuracy : {accuracy_val}")
    print()



  # Sauvegarde du modèle
  torch.save(model.state_dict(), "my_bert_exam_kk.pth")

if __name__=="__main__":
  main()