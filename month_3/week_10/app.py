
from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)
model = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
clf = pipeline('sentiment-analysis', model=model)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    out = clf(text)
    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
