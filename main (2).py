"""
🛡️ Prompt Safety Classifier - Multi-Model Comparison

Compare base vs pre-trained toxicity models for prompt safety classification.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd

from config import MODELS, PRETRAINED_TOXICITY, DEVICE, print_model_registry
from data import load_and_prepare_dataset, tokenize_dataset
from model import load_model_and_tokenizer, evaluate_model, train_model, clear_gpu_memory
from visualize import plot_comparison_dashboard, plot_radar_chart
from inference import SafetyClassifier, get_checkpoint_path, DEMO_PROMPTS


def build_results_dataframe(results_before, results_after):
    """Build comparison dataframe from before/after results."""
    rows = []
    for model_name in results_after.keys():
        before = results_before.get(model_name, {})
        after = results_after[model_name]
        rows.append({
            'Model': model_name,
            'Type': 'Pre-trained Toxicity' if model_name in PRETRAINED_TOXICITY else 'Base Model',
            'F1 Before': before.get('f1', 0),
            'F1 After': after['f1'],
            'F1 Δ': after['f1'] - before.get('f1', 0),
            'AUC Before': before.get('auc_roc', 0),
            'AUC After': after['auc_roc'],
            'AUC Δ': after['auc_roc'] - before.get('auc_roc', 0),
            'Accuracy': after['accuracy'],
            'Precision': after['precision'],
            'Recall': after['recall'],
            'AUPRC': after['auprc'],
            'Latency (ms)': after['latency_ms']
        })
    return pd.DataFrame(rows).sort_values('F1 After', ascending=False)


def print_leaderboard(results_df):
    """Print formatted leaderboard."""
    print("\n" + "="*100)
    print("🏆 LEADERBOARD - SORTED BY F1 SCORE")
    print("="*100)
    cols = ['Model', 'Type', 'F1 Before', 'F1 After', 'F1 Δ', 'AUC After', 'Latency (ms)']
    print(results_df[cols].to_string(index=False))
    print("="*100)


def print_summary(results_df):
    """Print final summary with key findings."""
    print("\n" + "="*80)
    print("📋 FINAL SUMMARY")
    print("="*80)

    winner = results_df.iloc[0]
    print(f"\n🏆 WINNER: {winner['Model']}")
    print(f"   F1: {winner['F1 After']:.4f} | AUC: {winner['AUC After']:.4f} | Latency: {winner['Latency (ms)']:.2f}ms")

    fastest = results_df.loc[results_df['Latency (ms)'].idxmin()]
    print(f"\n⚡ FASTEST: {fastest['Model']} ({fastest['Latency (ms)']:.2f}ms)")

    biggest_gain = results_df.loc[results_df['F1 Δ'].idxmax()]
    print(f"\n📈 BIGGEST IMPROVEMENT: {biggest_gain['Model']} (+{biggest_gain['F1 Δ']:.4f})")

    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    base_avg = results_df[results_df['Type']=='Base Model']['F1 Δ'].mean()
    pretrained_avg = results_df[results_df['Type']=='Pre-trained Toxicity']['F1 Δ'].mean()
    print(f"• Base models improved by avg +{base_avg:.3f} F1")
    print(f"• Pre-trained toxicity models improved by avg +{pretrained_avg:.3f} F1")
    print("="*80)


def main():
    # Setup
    print(f"🖥️ Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print_model_registry()

    # Load data
    train_dataset, test_dataset = load_and_prepare_dataset()

    # Train and evaluate all models
    results_before, results_after, all_probs_data = {}, {}, {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {model_name}")
        print(f"{'='*60}")

        try:
            model, tokenizer = load_model_and_tokenizer(model_path)
            
            print("   📝 Tokenizing...")
            train_tok = tokenize_dataset(train_dataset, tokenizer)
            test_tok = tokenize_dataset(test_dataset, tokenizer)

            # Before fine-tuning
            print("   📊 Evaluating BEFORE...")
            metrics_before, labels, _ = evaluate_model(model, test_tok)
            results_before[model_name] = metrics_before
            print(f"      F1: {metrics_before['f1']:.4f} | AUC: {metrics_before['auc_roc']:.4f}")

            # Fine-tune
            print("   🔧 Fine-tuning...")
            output_dir = f"./models/{model_name.lower().replace('-', '_')}"
            model = train_model(model, train_tok, test_tok, tokenizer, output_dir)

            # After fine-tuning
            print("   📊 Evaluating AFTER...")
            metrics_after, labels, probs_after = evaluate_model(model, test_tok)
            results_after[model_name] = metrics_after
            all_probs_data[model_name] = {'labels': labels, 'probs': probs_after}
            print(f"      F1: {metrics_after['f1']:.4f} | AUC: {metrics_after['auc_roc']:.4f}")

            f1_imp = metrics_after['f1'] - metrics_before['f1']
            print(f"   ✅ F1 Improvement: +{f1_imp:.4f}")

            del model
            clear_gpu_memory()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    print("\n" + "="*60)
    print("🎉 All models processed!")
    print("="*60)

    # Build results
    results_df = build_results_dataframe(results_before, results_after)
    print_leaderboard(results_df)

    # Visualizations
    plot_comparison_dashboard(results_df, all_probs_data)
    plot_radar_chart(results_df)

    # Test best model
    best_model = results_df.iloc[0]['Model']
    checkpoint = get_checkpoint_path(best_model, len(train_dataset))
    
    print(f"\n🧪 TESTING: {best_model}")
    print("="*70)
    try:
        classifier = SafetyClassifier(checkpoint, MODELS[best_model])
        print(classifier.predict_batch(DEMO_PROMPTS).to_string(index=False))
    except Exception as e:
        print(f"   ⚠️ Could not load checkpoint: {e}")

    # Summary & export
    print_summary(results_df)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("✅ Results exported to 'model_comparison_results.csv'")

    return results_df


if __name__ == "__main__":
    main()
