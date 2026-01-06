import torch
from PIL import Image
from transformers import CLIPProcessor
from pathlib import Path
from tqdm import tqdm

from src.models import create_model
from src.utils import load_config

class HatefulMemesPredictor:
    """Inference class for hateful memes classification"""
    
    def __init__(self, checkpoint_path, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate checkpoint exists
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model on {self.device}...")
        
        # Load model
        self.model = create_model('fusion', self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained(
            self.config['model']['image_encoder']
        )
        
        print("✓ Model loaded successfully!")
    
    @torch.no_grad()
    def predict(self, image_path, caption=""):
        """Predict if a meme is hateful"""
        
        # Validate image path
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and process image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(self.device)
        
        # Prepare text
        text = caption if caption else "no text available"
        
        # Predict
        logit = self.model(pixel_values, [text])
        prob = torch.sigmoid(logit).item()
        
        # Determine label and confidence
        if prob >= 0.5:
            label = "hateful"
            confidence = prob
        else:
            label = "not_hateful"
            confidence = 1 - prob
        
        return {
            'label': label,
            'confidence': confidence,
            'probability': prob
        }
    
    def predict_batch(self, image_paths, captions=None):
        """Predict for multiple images with progress bar"""
        if captions is None:
            captions = [""] * len(image_paths)
        
        if len(image_paths) != len(captions):
            raise ValueError("Number of images and captions must match")
        
        results = []
        for img_path, caption in tqdm(zip(image_paths, captions), 
                                      total=len(image_paths),
                                      desc="Predicting"):
            try:
                result = self.predict(img_path, caption)
                result['image_path'] = str(img_path)
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'image_path': str(img_path),
                    'status': 'error',
                    'error': str(e),
                    'label': None,
                    'confidence': None
                }
            results.append(result)
        
        return results

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict hateful memes')
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--caption', default='', help='Optional caption')
    parser.add_argument('--checkpoint', default='checkpoints/best_fusion.pt', 
                       help='Model checkpoint')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = HatefulMemesPredictor(args.checkpoint)
        
        # Predict
        result = predictor.predict(args.image, args.caption)
        
        print(f"\n{'='*50}")
        print("PREDICTION RESULTS")
        print('='*50)
        print(f"Image: {args.image}")
        print(f"Label: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Raw Probability: {result['probability']:.4f}")
        print('='*50)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
