from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from .config import VECTORIZER_CONFIG

class TfidfVectorizer:
    def __init__(self, max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 1)):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            tokenizer=lambda x: x,
            lowercase=False,
            token_pattern=None
        )
    
    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self):
        return len(self.vectorizer.vocabulary_)
