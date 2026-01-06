import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import easyocr
from transformers import CLIPProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from tqdm import tqdm
import warnings

class MemeDataset(Dataset):
    """Dataset for hateful memes classification"""
    
    def __init__(self, df, image_processor, use_ocr=True, ocr_reader=None, is_training=False):
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.use_ocr = use_ocr
        self.ocr_reader = ocr_reader
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and process image - FIXED: Handle multiple extensions
        image = self._load_image(row['image_path'])
        
        image_inputs = self.image_processor(
            images=image, 
            return_tensors="pt",
            do_rescale=True
        )
        
        # Process text
        caption = str(row.get('caption', ''))
        
        # Extract OCR text if enabled
        if self.use_ocr and self.ocr_reader and Path(row['image_path']).exists():
            ocr_text = self._extract_ocr(row['image_path'])
            if ocr_text:
                text = f"{caption} [OCR] {ocr_text}".strip()
            else:
                text = caption
        else:
            text = caption
        
        # Handle empty text
        if not text or text.isspace():
            text = "no text available"
        
        # Get label
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'text': text,
            'label': label,
            'image_path': row['image_path']
        }
    
    def _load_image(self, image_path):
        """Load image with fallback for different extensions"""
        try:
            # Try original path first
            p = Path(image_path)
            if p.exists():
                return Image.open(p).convert('RGB')
            
            # Try different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_path = p.with_suffix(ext)
                if alt_path.exists():
                    return Image.open(alt_path).convert('RGB')
            
            # If nothing works, return black image
            warnings.warn(f"Image not found: {image_path}, using blank image")
            return Image.new('RGB', (224, 224), color='black')
            
        except Exception as e:
            warnings.warn(f"Failed to load {image_path}: {e}, using blank image")
            return Image.new('RGB', (224, 224), color='black')
    
    def _extract_ocr(self, image_path):
        """Extract text from image using OCR"""
        try:
            results = self.ocr_reader.readtext(str(image_path), detail=0)
            return ' '.join(results) if results else ""
        except Exception as e:
            return ""

def prepare_data(data_path, config, logger):
    """Load and prepare dataset"""
    logger.info(f"Loading data from {data_path}")
    
    # Load your dataset
    if Path(data_path).suffix == '.csv':
        df = pd.read_csv(data_path)
    elif Path(data_path).suffix == '.json':
        df = pd.read_json(data_path)
    else:
        raise ValueError("Data must be CSV or JSON format")
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Verify required columns
    required_cols = ['image_path', 'caption', 'label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df

def create_splits(df, config, logger):
    """Create train/val/test splits"""
    splits_file = Path(config['data']['splits_file'])
    
    if splits_file.exists():
        logger.info(f"Loading existing splits from {splits_file}")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        train_df = df.iloc[splits['train']].reset_index(drop=True)
        val_df = df.iloc[splits['val']].reset_index(drop=True)
        test_df = df.iloc[splits['test']].reset_index(drop=True)
    else:
        logger.info("Creating new splits")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=config['data']['val_ratio'] + config['data']['test_ratio'],
            stratify=df['label'],
            random_state=config['seed']
        )
        
        # Second split: val vs test
        val_size = config['data']['val_ratio'] / (config['data']['val_ratio'] + config['data']['test_ratio'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['label'],
            random_state=config['seed']
        )
        
        # Save splits
        splits = {
            'train': train_df.index.tolist(),
            'val': val_df.index.tolist(),
            'test': test_df.index.tolist()
        }
        
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Splits saved to {splits_file}")
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_dataloaders(train_df, val_df, test_df, config, logger):
    """Create data loaders for training, validation, and testing"""
    
    # Initialize processors
    image_processor = CLIPProcessor.from_pretrained(config['model']['image_encoder'])
    
    # Initialize OCR reader if needed
    ocr_reader = None
    if config['text']['use_ocr']:
        logger.info("Initializing OCR reader (this may take 30 seconds)...")
        ocr_reader = easyocr.Reader(
            config['text']['ocr_languages'], 
            gpu=torch.cuda.is_available(),
            verbose=False
        )
        logger.info("OCR reader initialized")
    else:
        logger.info("OCR disabled - using captions only")
    
    # Create datasets
    train_dataset = MemeDataset(train_df, image_processor, config['text']['use_ocr'], ocr_reader, is_training=True)
    val_dataset = MemeDataset(val_df, image_processor, config['text']['use_ocr'], ocr_reader, is_training=False)
    test_dataset = MemeDataset(test_df, image_processor, config['text']['use_ocr'], ocr_reader, is_training=False)
    
    # FIXED: num_workers=0 for Windows compatibility
    num_workers = 0 if torch.cuda.is_available() else 0
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: Train={len(train_loader)} batches, Val={len(val_loader)}, Test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader
