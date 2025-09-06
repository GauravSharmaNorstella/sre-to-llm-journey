
from transformers import pipeline

fill = pipeline('text-generation', model='distilgpt2')
print(fill('Site Reliability Engineering with LLMs will'))
