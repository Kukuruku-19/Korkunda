import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from .config import MODEL_CONFIG, TRAINING_CONFIG

class ResumeClassifier:
    def __init__(self, input_size=5000, hidden_sizes=[256, 128], 
                 num_classes=24, dropout_rate=0.3, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self):
        self.model = Sequential()
        
        self.model.add(Dense(self.hidden_sizes[0], activation='relu', 
                            input_dim=self.input_size))
        self.model.add(Dropout(self.dropout_rate))
        
        self.model.add(Dense(self.hidden_sizes[1], activation='relu'))
        self.model.add(Dropout(self.dropout_rate))
        
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        return self.model
    
    def compile_model(self):
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        return self.model
