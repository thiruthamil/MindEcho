# app/llm_client.py
import os
from openai import OpenAI

# Ollama uses an OpenAI-compatible API; key value is unused but required
os.environ.setdefault("OPENAI_API_KEY", "ollama-local")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="http://localhost:11434/v1"
)

def llm_chat(messages, model="llama3", temperature=0.7, max_tokens=400):
    """Send chat messages to the local Ollama model."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
