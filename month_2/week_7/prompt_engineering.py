
from transformers import pipeline

gen = pipeline('text-generation', model='distilgpt2', max_new_tokens=60)

prompt = "Summarize the incident: CPU spike on database node, mitigation steps included..."
print(gen(prompt))
