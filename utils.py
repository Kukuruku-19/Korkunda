import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

def decode_labels(encoded_labels, encoder):
    return encoder.inverse_transform(encoded_labels)

def labels_to_categorical(labels, num_classes):
    return to_categorical(labels, num_classes=num_classes)

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
    return weights
