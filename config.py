"""Model registry and configuration."""

import torch

MODELS = {
    # Base models (need full fine-tuning)
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'DeBERTa-v3': 'microsoft/deberta-v3-small',
    'DistilBERT': 'distilbert-base-uncased',
    # Pre-trained toxicity models
    'Toxic-BERT': 'unitary/toxic-bert',
    'RoBERTa-Toxicity': 's-nlp/roberta_toxicity_classifier',
}

PRETRAINED_TOXICITY = {'Toxic-BERT', 'RoBERTa-Toxicity'}

# Training config
TRAIN_CONFIG = {
    'max_length': 128,
    'train_batch_size': 16,
    'eval_batch_size': 32,
    'epochs': 2,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'logging_steps': 100,
}

# Dataset config
DATA_CONFIG = {
    'dataset_name': 'google/civil_comments',
    'toxicity_threshold': 0.5,
    'samples_per_class': 5000,
    'test_size': 0.2,
    'seed': 42,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_model_registry():
    print("📋 Models to evaluate:")
    for name, path in MODELS.items():
        tag = "[PRE-TRAINED TOXICITY]" if name in PRETRAINED_TOXICITY else "[BASE]"
        print(f"   {name}: {path} {tag}")
