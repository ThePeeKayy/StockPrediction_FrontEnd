from flask import Flask, request, jsonify
import requests
from transformers import DebertaV2Tokenizer, DebertaForSequenceClassification
import torch
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
torch.cuda.empty_cache()

tokenizer = DebertaV2Tokenizer.from_pretrained('./deberta_tokenizer')
model = DebertaForSequenceClassification.from_pretrained('./deberta_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

GEN_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
gen_headers = {
    "Authorization": f"Bearer {os.getenv('HF_API_KEY')}"
}

@app.route('/analyze', methods=['POST'])
def predict():
    data = request.json
    text = data.get('news')
    symbol = data.get('stock')
    combined_text = f"For stock symbol {symbol}, {text}"

    inputs = tokenizer(combined_text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
    outputs = model(**inputs)
    sentiment = 'positive' if outputs.logits.argmax(-1).item() == 1 else 'negative'
    print(f"Sentiment: {sentiment}")

    advice_input_text = (
        f"Stock Symbol: {symbol}\n"
        f"News Summary: {combined_text}\n"
        f"Sentiment Analysis: {sentiment}\n\n"
        f"Task: Based on the news and sentiment, provide a detailed recommendation on whether an investor should buy, hold, or sell the stock {symbol}. "
    )

 
    len_advice = len(advice_input_text)
    payload = {
        "inputs": advice_input_text
    }

    try:
        response = requests.post(GEN_API_URL, headers=gen_headers, json=payload)
        response.raise_for_status()
        generated_advice = response.json()[0]['generated_text']
        generated_advice = generated_advice[len_advice:-1]
    except Exception as e:
        print(f"Error during text generation: {e}")
        return jsonify({'error': str(e)})

    return jsonify({'sentiment': sentiment, 'advice': generated_advice})

if __name__ == '__main__':
    app.run(debug=True)
