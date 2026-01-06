# verify_memotion.py
import pandas as pd
from pathlib import Path

df = pd.read_csv('data/raw/memes.csv')
print(f"✓ Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"\nSample data:")
print(df.head())

print(f"\nChecking first 5 images...")
for path in df['image_path'].head(5):
    exists = "✓" if Path(path).exists() else "✗"
    print(f"{exists} {path}")
