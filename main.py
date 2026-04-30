# ==========================================
# IMPORTS
# ==========================================
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import AnalyzeRequest, ComposeResult
from sentiment import analyze_sentiment
from composer import compose_response

# ==========================================
# APP CONFIGURATION
# ==========================================
app = FastAPI(title="ToneCraft AI")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# ROUTING
# ==========================================
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/app")
def get_app():
    return FileResponse("static/app.html")

@app.post("/analyze", response_model=ComposeResult)
def analyze(req: AnalyzeRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    sentiment = analyze_sentiment(req.message)
    composed = compose_response(req.message, sentiment)
    
    return ComposeResult(
        sentiment=sentiment,
        response_options=composed["response_options"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}