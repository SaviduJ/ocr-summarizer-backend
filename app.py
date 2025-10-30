from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

app = Flask(__name__)

# Get API key from .env
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("Hugging Face API key not found in .env file")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "moonshotai/Kimi-K2-Instruct-0905",
        "messages": [
            {"role": "system", "content": "Summarize the following text in clear and concise language."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return jsonify({
            "error": "Failed to get response from Hugging Face",
            "details": response.text
        }), 500

    try:
        summary = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": "Unexpected response format", "details": str(e)}), 500

    return jsonify({"summary": summary})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
