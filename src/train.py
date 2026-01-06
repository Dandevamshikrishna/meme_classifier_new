import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
import sys

from src.utils import set_seed, load_config, setup_logging, EarlyStopping, count_parameters
from src.data import prepare_data, create_splits, create_dataloaders
from src.models import create_model

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, config, model_type):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        texts = batch['text']
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=config['training']['use_amp']):
            if model_type == 'fusion':
                logits = model(pixel_values, texts)
            elif model_type == 'image_only':
                logits = model(pixel_values)
            else:  # text_only
                logits = model(texts)
            
            loss = criterion(logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config, model_type):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        texts = batch['text']
        labels = batch['label'].to(device)
        
        # Forward pass
        if model_type == 'fusion':
            logits = model(pixel_values, texts)
        elif model_type == 'image_only':
            logits = model(pixel_values)
        else:  # text_only
            logits = model(texts)
        
        loss = criterion(logits, labels)
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, all_probs) * 100
    
    return avg_loss, accuracy, auc, all_labels, all_probs

def train_model(model_type, config, logger, data_path):
    """Main training function"""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_type.upper()} model")
    logger.info(f"{'='*50}\n")
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    df = prepare_data(data_path, config, logger)
    train_df, val_df, test_df = create_splits(df, config, logger)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, config, logger)
    
    # Create model
    model = create_model(model_type, config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function with class weights
    pos_weight = torch.tensor([config['loss']['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='max'
    )
    
    # Training loop
    best_auc = 0
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config, model_type
        )
        
        # Validate
        val_loss, val_acc, val_auc, _, _ = evaluate(
            model, val_loader, criterion, device, config, model_type
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.2f}%")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint_path = checkpoint_dir / f"best_{model_type}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model (AUC: {val_auc:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_auc):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"\nTraining completed. Best Val AUC: {best_auc:.2f}%")
    return checkpoint_dir / f"best_{model_type}.pt"

def main():
    # Load config
    config = load_config()
    set_seed(config['seed'])
    
    # Setup logging
    logger = setup_logging(config['paths']['logs'])
    
    # Data path - UPDATE THIS with your actual data path
    data_path = "data/raw/memes_mini.csv"  # or .json
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please provide your dataset in CSV or JSON format with columns:")
        logger.info("  - image_path: path to meme image")
        logger.info("  - caption: text caption (can be empty)")
        logger.info("  - label: 0 (not hateful) or 1 (hateful)")
        return
    
    # Train all models
    # models_to_train = ['image_only', 'text_only', 'fusion']

    models_to_train = ['fusion']  # For quicker testing, only train fusion model    
    for model_type in models_to_train:
        try:
            checkpoint_path = train_model(model_type, config, logger, data_path)
            logger.info(f"\n{model_type} checkpoint saved to: {checkpoint_path}\n")
        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
