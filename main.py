import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.vectorizer import TfidfVectorizer
from src.classifier import ResumeClassifier
from src.evaluator import ModelEvaluator
from src.utils import encode_labels, labels_to_categorical, save_pickle, load_pickle
from src.config import (DATASET_PATH, MODELS_DIR, PREPROCESSING_CONFIG,
                       VECTORIZER_CONFIG, MODEL_CONFIG, TRAINING_CONFIG)

def main():
    print("=" * 80)
    print("RESUME CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    print("\n1. Loading dataset...")
    data_loader = DataLoader(DATASET_PATH)
    data_loader.load_dataset()
    removed = data_loader.validate_data()
    print(f"   Removed {removed} invalid records")
    
    texts, labels = data_loader.extract_texts_labels()
    print(f"   Total samples: {len(texts)}")
    
    distribution = data_loader.get_category_distribution()
    print(f"   Number of categories: {len(distribution)}")
    
    print("\n2. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(texts, labels)
    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    print("\n3. Preprocessing texts...")
    preprocessor = TextPreprocessor(**PREPROCESSING_CONFIG)
    X_train_tokens = preprocessor.process_batch(X_train)
    X_val_tokens = preprocessor.process_batch(X_val)
    X_test_tokens = preprocessor.process_batch(X_test)
    print(f"   Average tokens per resume: {np.mean([len(tokens) for tokens in X_train_tokens]):.0f}")
    
    print("\n4. Vectorizing texts...")
    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    X_val_vec = vectorizer.transform(X_val_tokens)
    X_test_vec = vectorizer.transform(X_test_tokens)
    print(f"   Vocabulary size: {vectorizer.get_vocabulary_size()}")
    print(f"   Vector shape: {X_train_vec.shape}")
    
    print("\n5. Encoding labels...")
    y_train_enc, label_encoder = encode_labels(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    y_train_cat = labels_to_categorical(y_train_enc, MODEL_CONFIG['num_classes'])
    y_val_cat = labels_to_categorical(y_val_enc, MODEL_CONFIG['num_classes'])
    print(f"   Number of classes: {MODEL_CONFIG['num_classes']}")
    
    print("\n6. Converting to dense arrays...")
    X_train_dense = X_train_vec.toarray()
    X_val_dense = X_val_vec.toarray()
    X_test_dense = X_test_vec.toarray()
    
    print("\n7. Building model...")
    classifier = ResumeClassifier(**MODEL_CONFIG)
    classifier.build_model()
    classifier.compile_model()
    classifier.model.summary()
    
    print("\n8. Training model...")
    history = classifier.train(
        X_train_dense, y_train_cat,
        X_val_dense, y_val_cat,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        patience=TRAINING_CONFIG['patience']
    )
    
    print("\n9. Evaluating model...")
    y_pred = classifier.predict(X_test_dense)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_enc, y_pred)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n10. Generating visualizations...")
    cm = evaluator.calculate_confusion_matrix(y_test_enc, y_pred)
    evaluator.plot_confusion_matrix(cm, filename='confusion_matrix.png')
    evaluator.plot_training_history(history, filename='training_history.png')
    
    print("\n11. Saving results...")
    evaluator.save_metrics_to_csv(metrics, 'metrics.csv')
    
    per_class = evaluator.calculate_per_class_metrics(y_test_enc, y_pred)
    evaluator.save_classification_report(per_class, 'classification_report.csv')
    
    print("\n12. Saving model...")
    classifier.save_model(os.path.join(MODELS_DIR, 'model.h5'))
    save_pickle(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    save_pickle(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    save_pickle(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved to: {MODELS_DIR}")
    print(f"Results saved to: {os.path.join(os.path.dirname(__file__), 'results')}")

if __name__ == "__main__":
    import numpy as np
    main()
