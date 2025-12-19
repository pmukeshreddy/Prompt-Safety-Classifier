"""Inference utilities for prompt safety classification."""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DEVICE, TRAIN_CONFIG


class SafetyClassifier:
    """Wrapper for easy inference with trained models."""
    
    def __init__(self, model_path, tokenizer_path=None, device=DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(device)
        self.model.eval()

    def predict(self, text):
        """Predict safety of a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=TRAIN_CONFIG['max_length'],
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)
        
        is_unsafe = probs[0][1] > 0.5
        return {
            'text': text[:60] + ('...' if len(text) > 60 else ''),
            'prediction': 'UNSAFE 🚫' if is_unsafe else 'SAFE ✅',
            'confidence': f'{probs[0].max().item():.1%}'
        }

    def predict_batch(self, texts):
        """Predict safety for multiple texts."""
        return pd.DataFrame([self.predict(t) for t in texts])


def get_checkpoint_path(model_name, train_size, batch_size=16, epochs=2):
    """Construct checkpoint path from training params."""
    steps = train_size // batch_size * epochs
    return f"./models/{model_name.lower().replace('-', '_')}/checkpoint-{steps}"


# Demo prompts for testing
DEMO_PROMPTS = [
    "How do I bake chocolate chip cookies?",
    "You're an absolute moron, I hate you",
    "What's the capital of France?",
    "I want to kill someone",
    "Can you help me learn Python?",
    "Go die in a fire you worthless piece of trash",
]
