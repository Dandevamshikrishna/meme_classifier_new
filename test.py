import pandas as pd
from pathlib import Path
import shutil

def convert_memotion_dataset():
    """Convert Memotion dataset to hateful memes format"""
    
    # Load the labels file (adjust filename if different)
    label_files = list(Path('data/raw/memotion').glob('*.csv'))
    
    if not label_files:
        print("ERROR: No CSV file found in data/raw/memotion/")
        print("Available files:", list(Path('data/raw/memotion').iterdir()))
        return
    
    print(f"Reading labels from: {label_files[0]}")
    df = pd.read_csv(label_files[0])
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Map Memotion labels to binary (hateful/not_hateful)
    # Memotion has: overall_sentiment, humour, sarcasm, offensive, motivational
    
    data = []
    
    for idx, row in df.iterrows():
        # Get image path
        img_name = row.get('image_name', row.get('image', f"{idx}.jpg"))
        img_path = f"data/raw/memotion/images/{img_name}"
        
        # Check if image exists
        if not Path(img_path).exists():
            continue
        
        # Get text/caption
        caption = str(row.get('text_ocr', row.get('text', '')))
        
        # Map to binary label
        # Consider "offensive" or "very_negative" sentiment as hateful (1)
        # Everything else as not_hateful (0)
        
        offensive = row.get('offensive', 0)
        sentiment = str(row.get('overall_sentiment', '')).lower()
        
        if offensive == 1 or 'negative' in sentiment or 'offensive' in sentiment:
            label = 1  # Hateful
        else:
            label = 0  # Not hateful
        
        data.append({
            'image_path': img_path,
            'caption': caption,
            'label': label
        })
    
    # Create final DataFrame
    df_final = pd.DataFrame(data)
    
    # Balance the dataset if too imbalanced
    num_hateful = (df_final['label'] == 1).sum()
    num_not_hateful = (df_final['label'] == 0).sum()
    
    print(f"\nOriginal distribution:")
    print(f"  Hateful (1): {num_hateful}")
    print(f"  Not Hateful (0): {num_not_hateful}")
    
    # Undersample majority class for balance
    if num_not_hateful > num_hateful * 2:
        df_hateful = df_final[df_final['label'] == 1]
        df_not_hateful = df_final[df_final['label'] == 0].sample(n=num_hateful * 2, random_state=42)
        df_final = pd.concat([df_hateful, df_not_hateful]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df_final.to_csv('data/raw/memes.csv', index=False)
    
    print(f"\n Converted {len(df_final)} samples")
    print(f" Final distribution:")
    print(df_final['label'].value_counts())    
    return df_final

if __name__ == "__main__":
    convert_memotion_dataset()
