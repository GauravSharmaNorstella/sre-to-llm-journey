
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Dataset
dataset = load_dataset('imdb')

# Model & tokenizer
model_name = 'distilbert-base-uncased'
tok = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tok(batch['text'], padding='max_length', truncation=True)

tok_ds = dataset.map(tokenize, batched=True)

tok_ds = tok_ds.remove_columns(['text'])
tok_ds = tok_ds.rename_column('label', 'labels')

tok_ds.set_format('torch')

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

args = TrainingArguments(
    output_dir='outputs',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds['train'].shuffle(seed=42).select(range(2000)),
    eval_dataset=tok_ds['test'].shuffle(seed=42).select(range(1000)),
    compute_metrics=compute_metrics,
)

trainer.train()
print('Model trained. Check ./outputs')
