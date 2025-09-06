
from transformers import pipeline
from pathlib import Path

logs = Path(__file__).parent / 'sample_logs.txt'
text = logs.read_text()

summarizer = pipeline('summarization', model='sshleifer/tiny-mbart')
print(summarizer(text, max_length=60, min_length=5, do_sample=False))
