import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HYPERBOLIC_API_KEY")

if not api_key:
    print("ERROR: HYPERBOLIC_API_KEY not found in environment")
    exit(1)

print(f"Testing Hyperbolic API...")
print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "messages": [
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ],
    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": False
}

try:
    print("\nSending request to Hyperbolic API...")
    response = requests.post(
        "https://api.hyperbolic.xyz/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nSUCCESS!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"\nFull response:")
        print(result)
    else:
        print(f"\nERROR!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\nEXCEPTION: {e}")
