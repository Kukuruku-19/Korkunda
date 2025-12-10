import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from .config import PREPROCESSING_CONFIG

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, use_lemmatization=True, 
                 min_word_length=2, generate_bigrams=False):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        self.generate_bigrams = generate_bigrams
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def _convert_to_lowercase(self, text):
        return text.lower()
    
    def _remove_special_characters(self, text):
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def _tokenize(self, text):
        return word_tokenize(text)
    
    def _filter_by_length(self, tokens):
        return [token for token in tokens if len(token) >= self.min_word_length]
    
    def _remove_stopwords_method(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    
    def _lemmatize(self, tokens):
        pos_tags = pos_tag(tokens)
        lemmatized = []
        for token, tag in pos_tags:
            pos = self._get_wordnet_pos(tag)
            lemmatized.append(self.lemmatizer.lemmatize(token, pos=pos))
        return lemmatized
    
    def _generate_bigrams_method(self, tokens):
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
        return tokens + bigrams
    
    def process(self, text):
        text = self._convert_to_lowercase(text)
        text = self._remove_special_characters(text)
        tokens = self._tokenize(text)
        tokens = self._filter_by_length(tokens)
        
        if self.remove_stopwords:
            tokens = self._remove_stopwords_method(tokens)
        
        if self.use_lemmatization:
            tokens = self._lemmatize(tokens)
        
        if self.generate_bigrams:
            tokens = self._generate_bigrams_method(tokens)
        
        return tokens
    
    def process_batch(self, texts):
        return [self.process(text) for text in texts]
