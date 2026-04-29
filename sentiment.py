# ==========================================
# IMPORTS
# ==========================================
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "iDharshan/my_customer_satisfaction_model"
LABELS = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}

# ==========================================
# INITIALIZATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()

# ==========================================
# CORE LOGIC
# ==========================================
def analyze_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    label_id = probs.argmax().item()
    return {
        "label": LABELS[label_id],
        "confidence": round(probs[label_id].item(), 3),
        "scores": {LABELS[i]: round(probs[i].item(), 3) for i in range(3)}
    }