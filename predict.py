import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.classifier import ResumeClassifier
from src.utils import load_pickle
from src.config import MODELS_DIR, CATEGORIES

def predict_resume(resume_text, model_type='baseline'):
    preprocessor = load_pickle(os.path.join(MODELS_DIR, f'{model_type}_preprocessor.pkl'))
    vectorizer = load_pickle(os.path.join(MODELS_DIR, f'{model_type}_vectorizer.pkl'))
    label_encoder = load_pickle(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    classifier = ResumeClassifier()
    classifier.load_model(os.path.join(MODELS_DIR, f'{model_type}_model.h5'))
    
    tokens = preprocessor.process(resume_text)
    vec = vectorizer.transform([tokens])
    vec_dense = vec.toarray()
    
    pred_class = classifier.predict(vec_dense)[0]
    pred_proba = classifier.predict_proba(vec_dense)[0]
    
    category = label_encoder.inverse_transform([pred_class])[0]
    confidence = pred_proba[pred_class]
    
    top_5_idx = np.argsort(pred_proba)[-5:][::-1]
    top_5_categories = label_encoder.inverse_transform(top_5_idx)
    top_5_probs = pred_proba[top_5_idx]
    
    return {
        'category': category,
        'confidence': confidence,
        'top_5': list(zip(top_5_categories, top_5_probs))
    }

def predict_from_file(filepath, model_type='baseline'):
    with open(filepath, 'r', encoding='utf-8') as f:
        resume_text = f.read()
    
    result = predict_resume(resume_text, model_type)
    
    print(f"\nPrediction using {model_type} model:")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop 5 predictions:")
    for i, (cat, prob) in enumerate(result['top_5'], 1):
        print(f"{i}. {cat}: {prob:.4f}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict resume category')
    parser.add_argument('--text', type=str, help='Resume text')
    parser.add_argument('--file', type=str, help='Path to resume file')
    parser.add_argument('--model', type=str, choices=['baseline', 'modified'],
                       default='modified', help='Model to use')
    
    args = parser.parse_args()
    
    if args.file:
        predict_from_file(args.file, args.model)
    elif args.text:
        result = predict_resume(args.text, args.model)
        print(f"\nPrediction using {args.model} model:")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nTop 5 predictions:")
        for i, (cat, prob) in enumerate(result['top_5'], 1):
            print(f"{i}. {cat}: {prob:.4f}")
    else:
        sample_resume = """
        Senior Software Engineer with 5+ years of experience in machine learning 
        and data analysis. Expert in Python, TensorFlow, and scikit-learn. 
        Developed and deployed multiple ML models for production systems.
        Strong background in data preprocessing, feature engineering, and model optimization.
        """
        
        print("Demo: Classifying sample resume...")
        print("\nResume text:")
        print(sample_resume)
        
        result = predict_resume(sample_resume, args.model)
        print(f"\nPrediction using {args.model} model:")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nTop 5 predictions:")
        for i, (cat, prob) in enumerate(result['top_5'], 1):
            print(f"{i}. {cat}: {prob:.4f}")
