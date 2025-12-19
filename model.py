"""Model training and evaluation utilities."""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from config import DEVICE, TRAIN_CONFIG


def load_model_and_tokenizer(model_path):
    """Load tokenizer and model from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    return model, tokenizer


def evaluate_model(model, test_data, device=DEVICE):
    """Evaluate model and return metrics + inference time."""
    model.eval()
    model.to(device)
    dataloader = DataLoader(test_data, batch_size=TRAIN_CONFIG['eval_batch_size'])

    all_preds, all_probs, all_labels = [], [], []
    total_time = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_time += time.time() - start

            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0,
        'latency_ms': (total_time / len(test_data)) * 1000
    }

    prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
    metrics['auprc'] = auc(rec_curve, prec_curve)

    return metrics, all_labels, all_probs


def train_model(model, train_data, test_data, tokenizer, output_dir):
    """Fine-tune model with HuggingFace Trainer."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAIN_CONFIG['epochs'],
        per_device_train_batch_size=TRAIN_CONFIG['train_batch_size'],
        per_device_eval_batch_size=TRAIN_CONFIG['eval_batch_size'],
        warmup_steps=TRAIN_CONFIG['warmup_steps'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        logging_steps=TRAIN_CONFIG['logging_steps'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {'f1': f1_score(labels, preds), 'accuracy': accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model


def clear_gpu_memory():
    """Clear GPU cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
