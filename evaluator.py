import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
from .config import FIGURES_DIR, METRICS_DIR, CATEGORIES
import os

class ModelEvaluator:
    def __init__(self, category_names=None):
        self.category_names = category_names if category_names else CATEGORIES
    
    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        return metrics
    
    def calculate_per_class_metrics(self, y_true, y_pred):
        report = classification_report(
            y_true, y_pred,
            target_names=self.category_names,
            output_dict=True,
            zero_division=0
        )
        return report
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix', filename=None):
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.category_names,
                   yticklabels=self.category_names)
        plt.title(title, fontsize=16)
        plt.ylabel('True Category', fontsize=14)
        plt.xlabel('Predicted Category', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history, filename=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history.history['loss'], label='Training Loss', color='blue')
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, baseline_metrics, modified_metrics, filename=None):
        metrics_names = ['Accuracy', 'Macro F1-Score', 'Weighted F1-Score']
        baseline_values = [
            baseline_metrics['accuracy'],
            baseline_metrics['f1_macro'],
            baseline_metrics['f1_weighted']
        ]
        modified_values = [
            modified_metrics['accuracy'],
            modified_metrics['f1_macro'],
            modified_metrics['f1_weighted']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline Model', color='skyblue')
        bars2 = ax.bar(x + width/2, modified_values, width, label='Modified Model', color='orange')
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison: Baseline vs Modified Model')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0.7, 0.9])
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_to_csv(self, metrics, filename):
        df = pd.DataFrame([metrics])
        df.to_csv(os.path.join(METRICS_DIR, filename), index=False)
    
    def save_classification_report(self, report, filename):
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(METRICS_DIR, filename))
