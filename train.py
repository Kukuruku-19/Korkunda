import os
import sys
import numpy as np
from scipy.sparse import vstack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.vectorizer import TfidfVectorizer
from src.classifier import ResumeClassifier
from src.evaluator import ModelEvaluator
from src.utils import encode_labels, labels_to_categorical, save_pickle
from src.config import (DATASET_PATH, MODELS_DIR, PREPROCESSING_CONFIG, 
                       VECTORIZER_CONFIG, MODEL_CONFIG, TRAINING_CONFIG)

def train_baseline_model():
    print("=" * 80)
    print("TRAINING BASELINE MODEL (Unigrams only)")
    print("=" * 80)
    
    data_loader = DataLoader(DATASET_PATH)
    data_loader.load_dataset()
    removed = data_loader.validate_data()
    print(f"Removed {removed} invalid records")
    
    texts, labels = data_loader.extract_texts_labels()
    print(f"Total samples: {len(texts)}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(texts, labels)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    preprocessor_config = PREPROCESSING_CONFIG.copy()
    preprocessor_config['generate_bigrams'] = False
    preprocessor = TextPreprocessor(**preprocessor_config)
    
    print("\nPreprocessing texts...")
    X_train_tokens = preprocessor.process_batch(X_train)
    X_val_tokens = preprocessor.process_batch(X_val)
    X_test_tokens = preprocessor.process_batch(X_test)
    
    vectorizer_config = VECTORIZER_CONFIG.copy()
    vectorizer_config['ngram_range'] = (1, 1)
    vectorizer = TfidfVectorizer(**vectorizer_config)
    
    print("Vectorizing texts...")
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    X_val_vec = vectorizer.transform(X_val_tokens)
    X_test_vec = vectorizer.transform(X_test_tokens)
    
    print(f"Vocabulary size: {vectorizer.get_vocabulary_size()}")
    
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
    
    print("\nModel architecture:")
    classifier.model.summary()
    
    print("\nTraining model...")
    history = classifier.train(
        X_train_dense, y_train_cat,
        X_val_dense, y_val_cat,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        patience=TRAINING_CONFIG['patience']
    )
    
    print("\nEvaluating on test set...")
    y_pred = classifier.predict(X_test_dense)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_enc, y_pred)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    cm = evaluator.calculate_confusion_matrix(y_test_enc, y_pred)
    evaluator.plot_confusion_matrix(cm, 
                                   title='Confusion Matrix - Baseline Model',
                                   filename='baseline_confusion_matrix.png')
    
    evaluator.plot_training_history(history, 
                                   filename='baseline_training_history.png')
    
    evaluator.save_metrics_to_csv(metrics, 'baseline_metrics.csv')
    
    per_class = evaluator.calculate_per_class_metrics(y_test_enc, y_pred)
    evaluator.save_classification_report(per_class, 'baseline_classification_report.csv')
    
    classifier.save_model(os.path.join(MODELS_DIR, 'baseline_model.h5'))
    save_pickle(vectorizer, os.path.join(MODELS_DIR, 'baseline_vectorizer.pkl'))
    save_pickle(preprocessor, os.path.join(MODELS_DIR, 'baseline_preprocessor.pkl'))
    save_pickle(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    print("\nBaseline model training completed!")
    print(f"Model saved to {MODELS_DIR}")
    
    return metrics, history

def train_modified_model():
    print("=" * 80)
    print("TRAINING MODIFIED MODEL (Unigrams + Bigrams)")
    print("=" * 80)
    
    data_loader = DataLoader(DATASET_PATH)
    data_loader.load_dataset()
    removed = data_loader.validate_data()
    print(f"Removed {removed} invalid records")
    
    texts, labels = data_loader.extract_texts_labels()
    print(f"Total samples: {len(texts)}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(texts, labels)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    preprocessor_config = PREPROCESSING_CONFIG.copy()
    preprocessor_config['generate_bigrams'] = True
    preprocessor = TextPreprocessor(**preprocessor_config)
    
    print("\nPreprocessing texts...")
    X_train_tokens = preprocessor.process_batch(X_train)
    X_val_tokens = preprocessor.process_batch(X_val)
    X_test_tokens = preprocessor.process_batch(X_test)
    
    vectorizer_config = VECTORIZER_CONFIG.copy()
    vectorizer_config['ngram_range'] = (1, 2)
    vectorizer = TfidfVectorizer(**vectorizer_config)
    
    print("Vectorizing texts...")
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    X_val_vec = vectorizer.transform(X_val_tokens)
    X_test_vec = vectorizer.transform(X_test_tokens)
    
    print(f"Vocabulary size: {vectorizer.get_vocabulary_size()}")
    
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
    
    print("\nModel architecture:")
    classifier.model.summary()
    
    print("\nTraining model...")
    history = classifier.train(
        X_train_dense, y_train_cat,
        X_val_dense, y_val_cat,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        patience=TRAINING_CONFIG['patience']
    )
    
    print("\nEvaluating on test set...")
    y_pred = classifier.predict(X_test_dense)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_enc, y_pred)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    cm = evaluator.calculate_confusion_matrix(y_test_enc, y_pred)
    evaluator.plot_confusion_matrix(cm, 
                                   title='Confusion Matrix - Modified Model',
                                   filename='modified_confusion_matrix.png')
    
    evaluator.plot_training_history(history, 
                                   filename='modified_training_history.png')
    
    evaluator.save_metrics_to_csv(metrics, 'modified_metrics.csv')
    
    per_class = evaluator.calculate_per_class_metrics(y_test_enc, y_pred)
    evaluator.save_classification_report(per_class, 'modified_classification_report.csv')
    
    classifier.save_model(os.path.join(MODELS_DIR, 'modified_model.h5'))
    save_pickle(vectorizer, os.path.join(MODELS_DIR, 'modified_vectorizer.pkl'))
    save_pickle(preprocessor, os.path.join(MODELS_DIR, 'modified_preprocessor.pkl'))
    
    print("\nModified model training completed!")
    print(f"Model saved to {MODELS_DIR}")
    
    return metrics, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train resume classification model')
    parser.add_argument('--model', type=str, choices=['baseline', 'modified', 'both'], 
                       default='both', help='Which model to train')
    
    args = parser.parse_args()
    
    if args.model in ['baseline', 'both']:
        baseline_metrics, baseline_history = train_baseline_model()
        print("\n")
    
    if args.model in ['modified', 'both']:
        modified_metrics, modified_history = train_modified_model()
        print("\n")
    
    if args.model == 'both':
        print("=" * 80)
        print("COMPARING MODELS")
        print("=" * 80)
        
        evaluator = ModelEvaluator()
        evaluator.plot_metrics_comparison(baseline_metrics, modified_metrics,
                                        filename='models_comparison.png')
        
        print("\nBaseline Model:")
        for key, value in baseline_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nModified Model:")
        for key, value in modified_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nImprovement:")
        for key in baseline_metrics.keys():
            diff = modified_metrics[key] - baseline_metrics[key]
            pct = (diff / baseline_metrics[key]) * 100
            print(f"  {key}: {diff:+.4f} ({pct:+.2f}%)")
