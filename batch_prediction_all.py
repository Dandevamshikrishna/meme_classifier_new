import pandas as pd
from src.infer import HatefulMemesPredictor
from tqdm import tqdm

def predict_all_direct(csv_path, checkpoint='checkpoints/best_fusion.pt'):
    """Predict using model directly (faster than API)"""
    
    # Load predictor
    print("Loading model...")
    predictor = HatefulMemesPredictor(checkpoint)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Predicting {len(df)} images...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            pred = predictor.predict(row['image_path'], str(row.get('caption', '')))
            
            results.append({
                'image_path': row['image_path'],
                'true_label': row['label'],
                'predicted_label': pred['label'],
                'confidence': pred['confidence'],
                'correct': (pred['label'] == 'hateful' and row['label'] == 1) or 
                          (pred['label'] == 'not_hateful' and row['label'] == 0)
            })
        except Exception as e:
            print(f"Error on {row['image_path']}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('predictions_all.csv', index=False)
    
    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    print(f"\nDone! Accuracy: {accuracy:.2f}%")
    print(f"Saved to: predictions_all.csv")
    
    return results_df

if __name__ == "__main__":
    results = predict_all_direct('data/raw/memes.csv')
