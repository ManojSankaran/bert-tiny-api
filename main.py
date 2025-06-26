from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_NAME = "mrm8488/bert-tiny-mnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    label = data.get("label", "Book appointment")

    inputs = tokenizer(text, label, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    result = probs[0][2].item()

    return {"text": text, "label": label, "confidence": round(result, 3)}