import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

from src.utils import set_seed, load_config, setup_logging
from src.data import prepare_data, create_splits, create_dataloaders
from src.models import create_model

@torch.no_grad()
def predict(model, dataloader, device, config, model_type):
    """Generate predictions"""
    model.eval()
    all_probs = []
    all_labels = []
    all_paths = []
    
    # ADDED: Progress bar
    for batch in tqdm(dataloader, desc=f"Predicting {model_type}"):
        pixel_values = batch['pixel_values'].to(device)
        texts = batch['text']
        labels = batch['label']
        
        if model_type == 'fusion':
            logits = model(pixel_values, texts)
        elif model_type == 'image_only':
            logits = model(pixel_values)
        else:  # text_only
            logits = model(texts)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        all_paths.extend(batch['image_path'])
    
    return np.array(all_labels), np.array(all_probs), all_paths

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate all evaluation metrics in percentages"""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    prec, rec, f1 = prec * 100, rec * 100, f1 * 100
    
    # AUC
    auc = roc_auc_score(y_true, y_prob) * 100
    
    # Confusion matrix (normalized by true class)
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    tn_pct, fp_pct = cm[0, 0], cm[0, 1]
    fn_pct, tp_pct = cm[1, 0], cm[1, 1]
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': {
            'tn_pct': tn_pct,
            'fp_pct': fp_pct,
            'fn_pct': fn_pct,
            'tp_pct': tp_pct
        }
    }
    
    return metrics, cm

def plot_confusion_matrix(cm, model_name, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=['Not Hateful', 'Hateful'],
        yticklabels=['Not Hateful', 'Hateful'],
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title(f'Confusion Matrix - {model_name} (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def find_misclassified_examples(y_true, y_prob, paths, n=10):
    """Find most confident misclassifications"""
    y_pred = (y_prob >= 0.5).astype(int)
    misclassified_mask = y_true != y_pred
    
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # FIXED: Handle case with fewer misclassifications than n
    if len(misclassified_indices) == 0:
        return []
    
    n = min(n, len(misclassified_indices))  # Don't request more than available
    
    misclassified_probs = y_prob[misclassified_indices]
    misclassified_true = y_true[misclassified_indices]
    misclassified_pred = y_pred[misclassified_indices]
    
    # Sort by confidence (distance from 0.5)
    confidence = np.abs(misclassified_probs - 0.5)
    sorted_indices = np.argsort(confidence)[::-1][:n]
    
    examples = []
    for idx in sorted_indices:
        orig_idx = misclassified_indices[idx]
        examples.append({
            'image_path': paths[orig_idx],
            'true_label': int(misclassified_true[idx]),
            'predicted_label': int(misclassified_pred[idx]),
            'confidence': float(misclassified_probs[idx]),
            'true_label_name': 'hateful' if misclassified_true[idx] == 1 else 'not_hateful',
            'predicted_label_name': 'hateful' if misclassified_pred[idx] == 1 else 'not_hateful'
        })
    
    return examples

def evaluate_model(model_type, checkpoint_path, config, logger, data_path):
    """Evaluate a trained model"""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating {model_type.upper()} model")
    logger.info(f"{'='*50}\n")
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    df = prepare_data(data_path, config, logger)
    train_df, val_df, test_df = create_splits(df, config, logger)
    _, _, test_loader = create_dataloaders(train_df, val_df, test_df, config, logger)
    
    # Load model
    model = create_model(model_type, config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Generate predictions
    y_true, y_prob, paths = predict(model, test_loader, device, config, model_type)
    
    # Calculate metrics
    metrics, cm = calculate_metrics(y_true, y_prob)
    
    # Print metrics
    logger.info("\nTest Set Metrics (%):")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    logger.info(f"  Precision: {metrics['precision']:.2f}%")
    logger.info(f"  Recall:    {metrics['recall']:.2f}%")
    logger.info(f"  F1 Score:  {metrics['f1']:.2f}%")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.2f}%")
    
    logger.info("\nConfusion Matrix (%):")
    logger.info(f"  TN: {metrics['confusion_matrix']['tn_pct']:.2f}%  FP: {metrics['confusion_matrix']['fp_pct']:.2f}%")
    logger.info(f"  FN: {metrics['confusion_matrix']['fn_pct']:.2f}%  TP: {metrics['confusion_matrix']['tp_pct']:.2f}%")
    
    # Save results
    results_dir = Path(config['paths']['results']) / model_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, model_type.upper(), results_dir / 'confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_prob, model_type.upper(), results_dir / 'roc_curve.png')
    
    # Find misclassified examples
    misclassified = find_misclassified_examples(y_true, y_prob, paths, n=10)
    with open(results_dir / 'misclassified_examples.json', 'w') as f:
        json.dump(misclassified, f, indent=2)
    
    logger.info(f"\nResults saved to {results_dir}")
    logger.info(f"Found {len(misclassified)} misclassified examples")
    
    return metrics

def main():
    # Load config
    config = load_config()
    set_seed(config['seed'])
    
    # Setup logging
    logger = setup_logging(config['paths']['logs'])
    
    # Data path
    data_path = "data/raw/memes.csv"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Evaluate all models
    all_results = {}
    models = ['image_only', 'text_only', 'fusion']
    
    for model_type in models:
        checkpoint_path = Path(config['paths']['checkpoints']) / f"best_{model_type}.pt"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found for {model_type}: {checkpoint_path}")
            continue
        
        try:
            metrics = evaluate_model(model_type, checkpoint_path, config, logger, data_path)
            all_results[model_type] = metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)
        logger.info(f"{'Model':<20} {'Accuracy %':<12} {'Precision %':<13} {'Recall %':<10} {'F1 %':<10} {'ROC-AUC %':<10}")
        logger.info("-"*80)
        
        for model_type, metrics in all_results.items():
            logger.info(
                f"{model_type:<20} "
                f"{metrics['accuracy']:<12.2f} "
                f"{metrics['precision']:<13.2f} "
                f"{metrics['recall']:<10.2f} "
                f"{metrics['f1']:<10.2f} "
                f"{metrics['roc_auc']:<10.2f}"
            )
        
        # Save combined results
        results_path = Path(config['paths']['results']) / 'all_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\nAll results saved to {results_path}")
    else:
        logger.error("No models were evaluated. Train models first with: python -m src.train")

if __name__ == "__main__":
    main()
