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

Customer Message: "{customer_message}"

Provide exactly 3 distinct responses. Format your output strictly as follows:

[OPTION 1]
(Write the first response here)

[OPTION 2]
(Write the second response here)

[OPTION 3]
(Write the third response here)
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
    )
    
    content = response.choices[0].message.content
    
    options = []
    current_option = []
    
    for line in content.split('\n'):
        if line.strip().startswith('[OPTION'):
            if current_option:
                options.append('\n'.join(current_option).strip())
                current_option = []
        elif line.strip() or current_option:
            current_option.append(line)
            
    if current_option:
        options.append('\n'.join(current_option).strip())
        
    if len(options) < 3:
        fallback_options = [opt.strip() for opt in content.split('\n\n') if opt.strip()]
        if len(fallback_options) >= 3:
            options = fallback_options[:3]
        else:
            options = [content]
            
    while len(options) > 3:
        options.pop()
    while len(options) < 3:
        options.append("Failed to generate additional option.")
        
    return {
        "response_options": options
    }