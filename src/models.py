import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPVisionModel
from sentence_transformers import SentenceTransformer

class ImageOnlyModel(nn.Module):
    """Image-only baseline classifier"""
    
    def __init__(self, encoder_name, dropout=0.3):
        super().__init__()
        self.encoder = CLIPModel.from_pretrained(encoder_name).vision_model
        
        # Freeze early layers (optional)
        for param in list(self.encoder.parameters())[:-10]:
            param.requires_grad = False
        
        embed_dim = 768  # CLIP ViT-B/32
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, pixel_values):
        img_features = self.encoder(pixel_values).pooler_output
        logits = self.classifier(img_features)
        return logits.squeeze(-1)

class TextOnlyModel(nn.Module):
    """Text-only baseline classifier"""
    
    def __init__(self, encoder_name, dropout=0.3):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        embed_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, texts):
        txt_features = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        logits = self.classifier(txt_features)
        return logits.squeeze(-1)

class FusionModel(nn.Module):
    """Multimodal fusion model (late fusion)"""
    
    def __init__(self, image_encoder_name, text_encoder_name, dropout=0.3):
        super().__init__()
        
        # Image encoder
        self.image_encoder = CLIPModel.from_pretrained(image_encoder_name).vision_model
        
        # Freeze early layers
        for param in list(self.image_encoder.parameters())[:-10]:
            param.requires_grad = False
        
        # Text encoder
        self.text_encoder = SentenceTransformer(text_encoder_name)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Dimensions
        img_dim = 768  # CLIP ViT-B/32
        txt_dim = self.text_encoder.get_sentence_embedding_dimension()
        fusion_dim = img_dim + txt_dim
        
        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, pixel_values, texts):
        # Extract image features
        img_features = self.image_encoder(pixel_values).pooler_output
        
        # Extract text features
        txt_features = self.text_encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Concatenate features
        fused_features = torch.cat([img_features, txt_features], dim=1)
        
        # Classify
        logits = self.fusion_head(fused_features)
        return logits.squeeze(-1)

def create_model(model_type, config):
    """Factory function to create models"""
    if model_type == 'image_only':
        model = ImageOnlyModel(
            config['model']['image_encoder'],
            config['model']['dropout']
        )
    elif model_type == 'text_only':
        model = TextOnlyModel(
            config['model']['text_encoder'],
            config['model']['dropout']
        )
    elif model_type == 'fusion':
        model = FusionModel(
            config['model']['image_encoder'],
            config['model']['text_encoder'],
            config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
