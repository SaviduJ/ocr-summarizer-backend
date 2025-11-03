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

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    payload = {
        "model": "moonshotai/Kimi-K2-Instruct-0905",
        "messages": [
            {"role": "system", "content": "Summarize the following text in clear and concise language."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(HF_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        return jsonify({"error": "Failed to get response from Hugging Face", "details": response.text}), 500

    try:
        summary = response.json()["choices"][0]["message"]["content"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": "Unexpected response format", "details": str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    payload = {
        "model": "moonshotai/Kimi-K2-Instruct-0905",
        "messages": [
            {"role": "system", "content": "Explain the following text in simple and detailed terms so anyone can understand it."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 800,
        "temperature": 0.8
    }

    response = requests.post(HF_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        return jsonify({"error": "Failed to get response from Hugging Face", "details": response.text}), 500

    try:
        explanation = response.json()["choices"][0]["message"]["content"]
        return jsonify({"explain": explanation})
    except Exception as e:
        return jsonify({"error": "Unexpected response format", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
