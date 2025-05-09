# groq_llm.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  

def ask_llama3(context, question):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a helpful assistant specialized in answering based only on the provided context.
If the answer is not found in the context, respond strictly with: "I cannot answer such questions."

Think step-by-step:
1. Understand the question.
2. Locate the relevant details in the context.
3. Generate a concise and factual response.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:
"""

    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    # Optional: fallback check
    fallback_message = "I cannot answer such questions."
    if not answer or fallback_message.lower() in answer.lower():
        return fallback_message

    return answer














