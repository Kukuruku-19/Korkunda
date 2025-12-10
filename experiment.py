import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.vectorizer import TfidfVectorizer
from src.classifier import ResumeClassifier
from src.evaluator import ModelEvaluator
from src.utils import encode_labels, labels_to_categorical
from src.config import DATASET_PATH, METRICS_DIR, MODEL_CONFIG, TRAINING_CONFIG

def run_experiment(config_name, preprocessing_config, vectorizer_config):
    print(f"\nRunning experiment: {config_name}")
    print("-" * 80)
    
    data_loader = DataLoader(DATASET_PATH)
    data_loader.load_dataset()
    data_loader.validate_data()
    
    texts, labels = data_loader.extract_texts_labels()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(texts, labels)
    
    preprocessor = TextPreprocessor(**preprocessing_config)
    X_train_tokens = preprocessor.process_batch(X_train)
    X_val_tokens = preprocessor.process_batch(X_val)
    X_test_tokens = preprocessor.process_batch(X_test)
    
    vectorizer = TfidfVectorizer(**vectorizer_config)
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    X_val_vec = vectorizer.transform(X_val_tokens)
    X_test_vec = vectorizer.transform(X_test_tokens)
    
    y_train_enc, label_encoder = encode_labels(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    y_train_cat = labels_to_categorical(y_train_enc, MODEL_CONFIG['num_classes'])
    y_val_cat = labels_to_categorical(y_val_enc, MODEL_CONFIG['num_classes'])
    
    X_train_dense = X_train_vec.toarray()
    X_val_dense = X_val_vec.toarray()
    X_test_dense = X_test_vec.toarray()
    
    classifier = ResumeClassifier(**MODEL_CONFIG)
    classifier.build_model()
    classifier.compile_model()
    
    history = classifier.train(
        X_train_dense, y_train_cat,
        X_val_dense, y_val_cat,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        patience=TRAINING_CONFIG['patience']
    )
    
    y_pred = classifier.predict(X_test_dense)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_enc, y_pred)
    
    metrics['config_name'] = config_name
    metrics['vocab_size'] = vectorizer.get_vocabulary_size()
    metrics['num_epochs'] = len(history.history['loss'])
    
    print(f"Results for {config_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"  Vocabulary size: {metrics['vocab_size']}")
    print(f"  Training epochs: {metrics['num_epochs']}")
    
    return metrics

def compare_configurations():
    print("=" * 80)
    print("CONFIGURATION COMPARISON EXPERIMENT")
    print("=" * 80)
    
    experiments = [
        {
            'name': 'Baseline (Unigrams only)',
            'preprocessing': {
                'remove_stopwords': True,
                'use_lemmatization': True,
                'min_word_length': 2,
                'generate_bigrams': False
            },
            'vectorizer': {
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 1)
            }
        },
        {
            'name': 'Modified (Unigrams + Bigrams)',
            'preprocessing': {
                'remove_stopwords': True,
                'use_lemmatization': True,
                'min_word_length': 2,
                'generate_bigrams': True
            },
            'vectorizer': {
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 2)
            }
        },
        {
            'name': 'No Stopwords Removal',
            'preprocessing': {
                'remove_stopwords': False,
                'use_lemmatization': True,
                'min_word_length': 2,
                'generate_bigrams': False
            },
            'vectorizer': {
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 1)
            }
        },
        {
            'name': 'No Lemmatization',
            'preprocessing': {
                'remove_stopwords': True,
                'use_lemmatization': False,
                'min_word_length': 2,
                'generate_bigrams': False
            },
            'vectorizer': {
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 1)
            }
        },
    ]
    
    results = []
    
    for exp in experiments:
        metrics = run_experiment(exp['name'], exp['preprocessing'], exp['vectorizer'])
        results.append(metrics)
    
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results[['config_name', 'accuracy', 'f1_macro', 'f1_weighted', 
                             'vocab_size', 'num_epochs']]
    
    print("\n", df_results.to_string(index=False))
    
    df_results.to_csv(os.path.join(METRICS_DIR, 'experiment_comparison.csv'), index=False)
    print(f"\nResults saved to {METRICS_DIR}/experiment_comparison.csv")
    
    best_idx = df_results['accuracy'].idxmax()
    best_config = df_results.loc[best_idx, 'config_name']
    best_accuracy = df_results.loc[best_idx, 'accuracy']
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    compare_configurations()
