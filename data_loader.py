import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import CATEGORIES, TRAINING_CONFIG

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.texts = None
        self.labels = None
        
    def load_dataset(self):
        self.data = pd.read_csv(self.dataset_path, encoding='utf-8')
        return self.data
    
    def validate_data(self):
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['Resume_str', 'Category'])
        self.data = self.data[self.data['Category'].isin(CATEGORIES)]
        removed_count = initial_count - len(self.data)
        return removed_count
    
    def extract_texts_labels(self):
        self.texts = self.data['Resume_str'].values
        self.labels = self.data['Category'].values
        return self.texts, self.labels
    
    def split_data(self, texts, labels):
        test_size = TRAINING_CONFIG['test_split']
        val_size = TRAINING_CONFIG['validation_split']
        random_state = TRAINING_CONFIG['random_state']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_category_distribution(self):
        return self.data['Category'].value_counts().to_dict()
