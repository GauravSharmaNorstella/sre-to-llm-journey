
from transformers import AutoTokenizer, AutoModel
import torch

name = 'sentence-transformers/all-MiniLM-L6-v2'
# If the above is heavy, you can switch to 'distilbert-base-uncased'

tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

text = ["SRE uses LLMs for incident analysis", "DevOps automation with AI"]
inputs = tok(text, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

# Simple embedding: mean-pool
emb = outputs.last_hidden_state.mean(dim=1)
print('Embeddings shape:', emb.shape)
