from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()


class PredictRequest(BaseModel):
    text: str = ""
    label: str = "Book appointment"

# Public MNLI model. Class index 2 is entailment, which /predict reports as confidence.
# The previous id, mrm8488/bert-tiny-mnli, is not on the Hugging Face Hub.
MODEL_NAME = "valhalla/distilbart-mnli-12-1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.post("/predict")
async def predict(body: PredictRequest):
    text = body.text
    label = body.label

    inputs = tokenizer(text, label, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    result = probs[0][2].item()

    return {"text": text, "label": label, "confidence": round(result, 3)}