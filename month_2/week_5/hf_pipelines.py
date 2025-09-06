
from transformers import pipeline
import os

model = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
clf = pipeline('sentiment-analysis', model=model)

print(clf('LLMs are transforming DevOps workflows!'))
