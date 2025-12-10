import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'Resume.csv')

PREPROCESSING_CONFIG = {
    'remove_stopwords': True,
    'use_lemmatization': True,
    'min_word_length': 2,
    'generate_bigrams': False
}

VECTORIZER_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.8,
    'ngram_range': (1, 1)
}

MODEL_CONFIG = {
    'input_size': 5000,
    'hidden_sizes': [256, 128],
    'num_classes': 24,
    'dropout_rate': 0.3,
    'learning_rate': 0.001
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.15,
    'test_split': 0.15,
    'patience': 10,
    'random_state': 42
}

CATEGORIES = [
    'HR', 'Designer', 'Information-Technology', 'Teacher', 'Advocate',
    'Business-Development', 'Healthcare', 'Fitness', 'Agriculture', 'BPO',
    'Sales', 'Consultant', 'Digital-Media', 'Automobile', 'Chef',
    'Finance', 'Apparel', 'Engineering', 'Accountant', 'Construction',
    'Public-Relations', 'Banking', 'Arts', 'Aviation'
]
