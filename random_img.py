import pandas as pd

# Load your full dataset
df = pd.read_csv('data/raw/memes.csv')

# Take small balanced sample
df_hateful = df[df['label'] == 1].sample(n=100, random_state=42)
df_not_hateful = df[df['label'] == 0].sample(n=100, random_state=42)

mini_df = pd.concat([df_hateful, df_not_hateful]).sample(frac=1, random_state=42)

# Save mini dataset
mini_df.to_csv('data/raw/memes_mini.csv', index=False)
print(f"✓ Created mini dataset: {len(mini_df)} samples")
print(f"Class balance: {mini_df['label'].value_counts().to_dict()}")
