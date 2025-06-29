import requests
import json
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"   # replace with your local model name

def ollama_chat(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
    }
    resp = requests.post(OLLAMA_URL, json=payload, stream=True)
    final_response = ""
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line.decode())
            final_response += chunk["response"]
    # Clean up any newlines, spaces
    
    return final_response
