hateful-memes-mm/
├── data/
│   ├── raw/                  # Original dataset (memes.csv, images/)
│   ├── processed/             # Cleaned dataset
│   └── splits.json            # Stratified dataset splits
├── src/
│   ├── data.py               # Dataset loader + OCR (EasyOCR)
│   ├── models.py             # Image/Text/Fusion architectures
│   ├── train.py              # Training loop with AMP + early stopping
│   ├── eval.py               # Test metrics (Accuracy, Precision, F1, ROC-AUC)
│   ├── infer.py              # Inference utilities (single/batch)
│   └── utils.py              # Helper functions (seeding, configs, logging)
├── checkpoints/               # Trained models (.pt files)
├── results/                   # Metrics + confusion matrices + ROC curves
├── app.py                     # FastAPI inference service
├── config.yaml                # Hyperparameters
├── requirements.txt           # Dependencies
├── Makefile                   # One-command utilities
├── report.md                  # Full experiment documentation
└── README.md                  # You're reading this!


1: Create Virtual Environment

python -m venv venv
venv\Scripts\activate       # Windows
# OR
source venv/bin/activate    # Linux/Mac

2; Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

3.Memotion Dataset (7K memes) from Kaggle

https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

4.Train the image-only, text-only, and fusion models:
    
    python -m src.train

5. Evaluation
   
    python -m src.eval

6. Inference (Single Meme Prediction)

 python -m src.infer --image data/raw/memotion/images/image_1640.jpg --caption "test"

7.FastAPI Deployment
    
    python app.py
  Open in browser:
    http://localhost:8000/docs

8. Batch Prediction (All Images)
    Predict all meme images at once and save results:
    python batch_prediction_all.py
