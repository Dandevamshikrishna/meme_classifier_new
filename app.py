from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from pathlib import Path
import uvicorn

from src.infer import HatefulMemesPredictor
from src.utils import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Hateful Memes Classifier API",
    description="Multimodal classification of hateful memes using image and text",
    version="1.0.0"
)

# Global predictor (loaded on startup)
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    config = load_config()
    checkpoint_path = Path(config['paths']['checkpoints']) / 'best_fusion.pt'
    
    if not checkpoint_path.exists():
        raise RuntimeError(f"Model checkpoint not found: {checkpoint_path}")
    
    predictor = HatefulMemesPredictor(str(checkpoint_path))
    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hateful Memes Classifier API",
        "endpoints": {
            "/predict": "POST - Classify a meme image",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Meme image file"),
    caption: str = Form(default="", description="Optional caption text")
):
    """
    Predict if a meme is hateful or not
    
    Args:
        image: Uploaded image file (JPG, PNG, etc.)
        caption: Optional text caption
    
    Returns:
        JSON with label and confidence
    """
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(image.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read and validate image
        image_bytes = await image.read()
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save temporarily for prediction
        temp_path = Path("temp_image.jpg")
        pil_image.save(temp_path)
        
        # Predict
        result = predictor.predict(temp_path, caption)
        
        # Clean up
        temp_path.unlink()
        
        return JSONResponse(content={
            "label": result['label'],
            "confidence": round(result['confidence'], 4)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
