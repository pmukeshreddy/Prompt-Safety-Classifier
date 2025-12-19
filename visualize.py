"""Visualization utilities for model comparison."""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from sklearn.metrics import roc_curve
from config import PRETRAINED_TOXICITY


def plot_comparison_dashboard(results_df, all_probs_data, save_path='multi_model_comparison.png'):
    """Generate 6-panel comparison dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🛡️ Prompt Safety Classifier - Multi-Model Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
    model_names = results_df['Model'].tolist()
    x = np.arange(len(model_names))
    width = 0.35

    # 1. F1 Before vs After
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, results_df['F1 Before'], width, label='Before', color='#ff6b6b', alpha=0.8)
    ax1.bar(x + width/2, results_df['F1 After'], width, label='After', color='#4ecdc4', alpha=0.8)
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score: Before vs After')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # 2. F1 Leaderboard
    ax2 = axes[0, 1]
    bars = ax2.barh(model_names[::-1], results_df['F1 After'].values[::-1], color=colors[::-1])
    ax2.set_xlabel('F1 Score')
    ax2.set_title('🏆 F1 Score Leaderboard')
    ax2.set_xlim(0, 1)
    for bar, val in zip(bars, results_df['F1 After'].values[::-1]):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

    # 3. AUC-ROC Comparison
    ax3 = axes[0, 2]
    ax3.bar(x - width/2, results_df['AUC Before'], width, label='Before', color='#ff6b6b', alpha=0.8)
    ax3.bar(x + width/2, results_df['AUC After'], width, label='After', color='#4ecdc4', alpha=0.8)
    ax3.set_ylabel('AUC-ROC')
    ax3.set_title('AUC-ROC: Before vs After')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1)

    # 4. ROC Curves
    ax4 = axes[1, 0]
    for model_name in model_names:
        if model_name in all_probs_data:
            fpr, tpr, _ = roc_curve(all_probs_data[model_name]['labels'], all_probs_data[model_name]['probs'])
            auc_val = results_df[results_df['Model'] == model_name]['AUC After'].values[0]
            ax4.plot(fpr, tpr, label=f'{model_name} ({auc_val:.3f})', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves (After Fine-tuning)')
    ax4.legend(loc='lower right', fontsize=8)

    # 5. Improvement by Model Type
    ax5 = axes[1, 1]
    base_models = results_df[results_df['Type'] == 'Base Model']['F1 Δ'].mean()
    pretrained = results_df[results_df['Type'] == 'Pre-trained Toxicity']['F1 Δ'].mean()
    ax5.bar(['Base Models', 'Pre-trained Toxicity'], [base_models, pretrained],
            color=['#3498db', '#9b59b6'], alpha=0.8)
    ax5.set_ylabel('Average F1 Improvement')
    ax5.set_title('Avg Improvement by Model Type')
    for i, v in enumerate([base_models, pretrained]):
        ax5.text(i, v + 0.01, f'+{v:.3f}', ha='center', fontweight='bold')

    # 6. Latency vs Performance
    ax6 = axes[1, 2]
    ax6.scatter(results_df['Latency (ms)'], results_df['F1 After'],
                c=range(len(results_df)), cmap='Set2', s=200, alpha=0.8)
    for i, name in enumerate(model_names):
        ax6.annotate(name, (results_df.iloc[i]['Latency (ms)'], results_df.iloc[i]['F1 After']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax6.set_xlabel('Latency (ms/sample)')
    ax6.set_ylabel('F1 Score')
    ax6.set_title('Performance vs Speed Trade-off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved to '{save_path}'")


def plot_radar_chart(results_df, save_path='radar_comparison.png'):
    """Radar chart comparing top 4 models across metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 After', 'AUC After', 'AUPRC']
    top_models = results_df.head(4)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    colors_radar = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for idx, (_, row) in enumerate(top_models.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Top 4 Models - Multi-Metric Radar', size=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved to '{save_path}'")
