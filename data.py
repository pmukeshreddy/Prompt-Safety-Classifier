"""Dataset loading and preprocessing."""

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from config import DATA_CONFIG


def create_binary_label(example):
    """Convert continuous toxicity score to binary label."""
    example['label'] = 1 if example['toxicity'] >= DATA_CONFIG['toxicity_threshold'] else 0
    return example


def load_and_prepare_dataset():
    """Load Civil Comments and create balanced train/test splits."""
    print("📥 Loading Civil Comments dataset...")
    dataset = load_dataset(DATA_CONFIG['dataset_name'], split="train")
    dataset = dataset.map(create_binary_label)

    # Balance dataset
    toxic = dataset.filter(lambda x: x['label'] == 1).shuffle(seed=DATA_CONFIG['seed'])
    non_toxic = dataset.filter(lambda x: x['label'] == 0).shuffle(seed=DATA_CONFIG['seed'])
    
    n_samples = DATA_CONFIG['samples_per_class']
    toxic = toxic.select(range(min(n_samples, len(toxic))))
    non_toxic = non_toxic.select(range(n_samples))
    
    balanced = concatenate_datasets([toxic, non_toxic]).shuffle(seed=DATA_CONFIG['seed'])

    # Split
    split = balanced.train_test_split(test_size=DATA_CONFIG['test_size'], seed=DATA_CONFIG['seed'])
    train_dataset, test_dataset = split['train'], split['test']

    print(f"✅ Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"   Label dist: {pd.Series([x['label'] for x in test_dataset]).value_counts().to_dict()}")
    
    return train_dataset, test_dataset


def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenize dataset for a specific tokenizer."""
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    cols_to_remove = [c for c in tokenized.column_names if c not in ['input_ids', 'attention_mask', 'label']]
    tokenized = tokenized.remove_columns(cols_to_remove)
    tokenized.set_format('torch')
    return tokenized
