# ToneCraft AI: Emotion-Aware Response Composer

ToneCraft AI is a privacy-first, locally-optimized customer experience tool. It analyzes customer message sentiment and generates empathetic, brand-aligned responses using a combination of local BERT models and the NVIDIA NIM API.

## Features
- **Emotion Analysis**: Local BERT-based sentiment detection.
- **AI Response Generation**: 3 distinct response options per query via NVIDIA NIM API.
- **Local Optimization**: Fast local inference for sentiment.
- **Glassmorphism UI**: Premium, modern interface with Light/Dark mode support.
- **Privacy-First**: No data sent to 3rd parties except for generation (via encrypted API).

## Tech Stack
- **Backend**: FastAPI, PyTorch, Transformers, OpenAI (for NVIDIA NIM).
- **Frontend**: Vanilla HTML/CSS/JS (Glassmorphism design).
- **Inference**: NVIDIA NIM API (Qwen/Llama-3.1).

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your `NVIDIA_API_KEY`.
4. Run the server: `uvicorn main:app --reload`
