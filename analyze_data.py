import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.config import DATASET_PATH, FIGURES_DIR, PREPROCESSING_CONFIG

def analyze_dataset():
    print("=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    data_loader = DataLoader(DATASET_PATH)
    data_loader.load_dataset()
    removed = data_loader.validate_data()
    
    print(f"\nRemoved records: {removed}")
    print(f"Valid records: {len(data_loader.data)}")
    
    distribution = data_loader.get_category_distribution()
    print(f"\nNumber of categories: {len(distribution)}")
    
    print("\nCategory distribution:")
    for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data_loader.data)) * 100
        print(f"  {category:25s}: {count:4d} ({percentage:5.2f}%)")
    
    texts, labels = data_loader.extract_texts_labels()
    
    text_lengths = [len(text) for text in texts]
    print(f"\nText length statistics (characters):")
    print(f"  Mean: {np.mean(text_lengths):.0f}")
    print(f"  Median: {np.median(text_lengths):.0f}")
    print(f"  Min: {np.min(text_lengths):.0f}")
    print(f"  Max: {np.max(text_lengths):.0f}")
    
    preprocessor = TextPreprocessor(**PREPROCESSING_CONFIG)
    
    print("\nProcessing sample texts...")
    sample_tokens = preprocessor.process_batch(texts[:100])
    token_counts = [len(tokens) for tokens in sample_tokens]
    
    print(f"\nToken count statistics (after preprocessing):")
    print(f"  Mean: {np.mean(token_counts):.0f}")
    print(f"  Median: {np.median(token_counts):.0f}")
    print(f"  Min: {np.min(token_counts):.0f}")
    print(f"  Max: {np.max(token_counts):.0f}")
    
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sorted_dist = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    axes[0, 0].barh(list(sorted_dist.keys()), list(sorted_dist.values()), color='skyblue')
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_title('Resume Distribution by Category')
    axes[0, 0].tick_params(axis='y', labelsize=8)
    
    axes[0, 1].hist(text_lengths, bins=30, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Text Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(token_counts, bins=20, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Token Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Token Counts (Sample)')
    axes[1, 0].grid(True, alpha=0.3)
    
    category_counts = list(sorted_dist.values())
    axes[1, 1].boxplot([category_counts], vert=False)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('Category Size Distribution')
    axes[1, 1].set_yticklabels(['All Categories'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {FIGURES_DIR}/dataset_analysis.png")
    
    df_dist = pd.DataFrame(list(distribution.items()), columns=['Category', 'Count'])
    df_dist['Percentage'] = (df_dist['Count'] / df_dist['Count'].sum() * 100).round(2)
    df_dist = df_dist.sort_values('Count', ascending=False)
    df_dist.to_csv(os.path.join(FIGURES_DIR.replace('figures', 'metrics'), 
                                'category_distribution.csv'), index=False)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    analyze_dataset()
