# ==========================================
# IMPORTS
# ==========================================
from pydantic import BaseModel

# ==========================================
# DATA MODELS
# ==========================================
class AnalyzeRequest(BaseModel):
    message: str

class SentimentResult(BaseModel):
    label: str
    confidence: float
    scores: dict

class ComposeResult(BaseModel):
    sentiment: SentimentResult
    response_options: list[str]