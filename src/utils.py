import random
import numpy as np
import torch
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file with complete defaults"""
    from pathlib import Path
    
    # Complete default configuration with ALL required keys
    default_config = {
        'seed': 42,
        'experiment_name': 'hateful_memes_multimodal',
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'splits_file': 'data/splits.json',
            'train_ratio': 0.7,
            'val_ratio': 0.1,
            'test_ratio': 0.2
        },
        'model': {
            'image_encoder': 'openai/clip-vit-base-patch32',
            'text_encoder': 'sentence-transformers/all-MiniLM-L6-v2',
            'image_embed_dim': 768,
            'text_embed_dim': 384,
            'hidden_dim': 256,
            'dropout': 0.3
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 30,
            'learning_rate': 0.0002,
            'weight_decay': 0.0001,
            'early_stopping_patience': 5,
            'use_amp': True,
            'gradient_clip': 1.0
        },
        'image': {
            'size': 224,
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711]
        },
        'text': {
            'max_length': 128,
            'use_ocr': True,
            'ocr_languages': ['en']
        },
        'loss': {
            'pos_weight': 1.5
        },
        'paths': {
            'checkpoints': 'checkpoints',
            'logs': 'logs',
            'results': 'results'
        },
        'device': 'cuda'
    }
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"⚠ Config file not found: {config_file.absolute()}")
        print("✓ Using default configuration")
        return default_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
        
        if loaded_config is None:
            print("⚠ Config file is empty")
            print("✓ Using default configuration")
            return default_config
        
        # Merge loaded config with defaults (loaded config takes precedence)
        def merge_dicts(default, loaded):
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_dicts(default_config, loaded_config)
        print("✓ Config loaded successfully from", config_file)
        return config
        
    except Exception as e:
        print(f"✗ Error reading config file: {e}")
        print("✓ Using default configuration")
        return default_config


def save_json(data, path):
    """Save data to JSON file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """Load data from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=5, mode='max', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.delta
        else:
            return score < self.best_score - self.delta
