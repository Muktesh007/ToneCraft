# ==========================================
# IMPORTS
# ==========================================
import os
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MODEL_ID = "meta/llama-3.1-70b-instruct"

TONE_GUIDE = {
    "Dissatisfied": "very empathetic, apologetic, solution-focused",
    "Neutral": "friendly, helpful, clear",
    "Satisfied": "warm, appreciative, positive"
}

# ==========================================
# INITIALIZATION
# ==========================================
print(f"Initializing NVIDIA NIM API for {MODEL_ID}...")
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ==========================================
# CORE LOGIC
# ==========================================
def compose_response(customer_message: str, sentiment: dict) -> dict:
    if not NVIDIA_API_KEY or NVIDIA_API_KEY == "nvapi-your-key-here":
        return {"response_options": ["ERROR: Please add your valid NVIDIA API key to the .env file.", "Failed to generate.", "Failed to generate."]}
        
    tone = TONE_GUIDE.get(sentiment["label"], "professional")
    
    prompt = f"""You are an expert customer support agent.
The customer's message has been analyzed and their mood is: {sentiment["label"]}.
Your task is to write exactly 3 distinct response options for the business owner to choose from.
The tone of all options should be: {tone}.

CRITICAL INSTRUCTION: If the customer's message is a general question, casual chat, or anything NOT related to customer support, complaints, reviews, or business inquiries, you MUST NOT generate responses. Instead, return exactly this fallback message for all 3 options:

[OPTION 1]
ToneCraft AI is designed specifically for customer support. Please provide a customer inquiry or review.

[OPTION 2]
I can only assist with responding to customer service related messages.

[OPTION 3]
Fallback: Message is not a customer concern.

If it IS a valid customer support message, provide exactly 3 distinct responses. Format your output strictly as follows:

[OPTION 1]
(Write the first response here)

[OPTION 2]
(Write the second response here)

[OPTION 3]
(Write the third response here)

Customer Message: "{customer_message}"
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
    )
    
    import re
    content = response.choices[0].message.content
    
    # Split by Option markers and clean up
    raw_options = re.split(r'\[OPTION \d+\]', content)
    options = [opt.strip() for opt in raw_options if opt.strip()]
    
    # Ensure we have exactly 3 options
    while len(options) > 3:
        options.pop()
    while len(options) < 3:
        options.append("Additional option could not be formatted.")
        
    return {
        "response_options": options
    }