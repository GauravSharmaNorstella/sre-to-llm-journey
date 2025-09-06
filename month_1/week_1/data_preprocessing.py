
import pandas as pd
from pathlib import Path

fp = Path(__file__).parent / 'sample.csv'
df = pd.read_csv(fp)

# Basic cleanup
df = df.dropna()
df['text'] = df['text'].str.strip().str.lower()

print('Rows:', len(df))
print(df.head())

# Save processed
out = Path(__file__).parent / 'processed.csv'
df.to_csv(out, index=False)
print('Saved ->', out)
