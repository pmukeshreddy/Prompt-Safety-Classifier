# 🛡️ Prompt Safety Classifier

Multi-model comparison for prompt safety/toxicity classification.

## Models Compared
- **Base Models**: BERT, RoBERTa, DeBERTa-v3, DistilBERT
- **Pre-trained Toxicity**: Toxic-BERT, RoBERTa-Toxicity

## Structure
```
prompt_safety_classifier/
├── config.py       # Model registry & hyperparameters
├── data.py         # Dataset loading & preprocessing
├── model.py        # Training & evaluation utilities
├── visualize.py    # Plotting functions
├── inference.py    # Prediction wrapper
├── main.py         # Orchestration script
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Outputs
- `model_comparison_results.csv` - Metrics for all models
- `multi_model_comparison.png` - 6-panel comparison dashboard
- `radar_comparison.png` - Radar chart for top 4 models

## Inference
```python
from inference import SafetyClassifier

classifier = SafetyClassifier("./models/deberta_v3/checkpoint-1000")
result = classifier.predict("Your text here")
```
