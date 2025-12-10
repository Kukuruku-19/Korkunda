import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.classifier import ResumeClassifier
from src.evaluator import ModelEvaluator
from src.utils import load_pickle, encode_labels
from src.config import DATASET_PATH, MODELS_DIR

def evaluate_model(model_type='baseline'):
    print(f"Evaluating {model_type} model...")
    
    data_loader = DataLoader(DATASET_PATH)
    data_loader.load_dataset()
    data_loader.validate_data()
    
    texts, labels = data_loader.extract_texts_labels()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(texts, labels)
    
    preprocessor = load_pickle(os.path.join(MODELS_DIR, f'{model_type}_preprocessor.pkl'))
    vectorizer = load_pickle(os.path.join(MODELS_DIR, f'{model_type}_vectorizer.pkl'))
    label_encoder = load_pickle(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    X_test_tokens = preprocessor.process_batch(X_test)
    X_test_vec = vectorizer.transform(X_test_tokens)
    X_test_dense = X_test_vec.toarray()
    
    y_test_enc = label_encoder.transform(y_test)
    
    classifier = ResumeClassifier()
    classifier.load_model(os.path.join(MODELS_DIR, f'{model_type}_model.h5'))
    
    y_pred = classifier.predict(X_test_dense)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_enc, y_pred)
    
    print(f"\n{model_type.upper()} Model Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    cm = evaluator.calculate_confusion_matrix(y_test_enc, y_pred)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    
    per_class = evaluator.calculate_per_class_metrics(y_test_enc, y_pred)
    
    return metrics, cm, per_class

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, choices=['baseline', 'modified', 'both'],
                       default='both', help='Which model to evaluate')
    
    args = parser.parse_args()
    
    if args.model in ['baseline', 'both']:
        baseline_metrics, baseline_cm, baseline_per_class = evaluate_model('baseline')
    
    if args.model in ['modified', 'both']:
        modified_metrics, modified_cm, modified_per_class = evaluate_model('modified')
    
    if args.model == 'both':
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        
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
