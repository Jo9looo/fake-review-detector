import sys
from types import ModuleType

# Mock ONLY the compiled binary module that is blocked by AppLocker
dummy_group = ModuleType('scipy.optimize._group_columns')
dummy_group.group_dense = lambda *args, **kwargs: None
dummy_group.group_sparse = lambda *args, **kwargs: None
sys.modules['scipy.optimize._group_columns'] = dummy_group

import os
import json
import joblib
import numpy as np

def extract_and_save():
    model_dir = 'outputs/results'
    model_path = os.path.join(model_dir, 'best_model_combined.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    
    print("Loading joblib files...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    tfidf = joblib.load(tfidf_path)
    
    print("Extracting SVM weights...")
    # coef_ is a sparse or dense matrix of shape (1, n_features)
    if hasattr(model, 'coef_'):
        if hasattr(model.coef_, 'toarray'):
            coef = model.coef_.toarray()[0].tolist()
        else:
            coef = model.coef_[0].tolist()
    else:
        raise ValueError("Model does not have coef_ attribute. Make sure it's a linear model.")
        
    intercept = float(model.intercept_[0])
    
    print("Extracting Scaler parameters...")
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    
    print("Extracting TF-IDF parameters...")
    # Vocabulary mapping: word -> index (converted to standard Python int for JSON serialization)
    vocab = {word: int(idx) for word, idx in tfidf.vocabulary_.items()}
    # IDF values: array of shape (n_features,)
    idf = tfidf.idf_.tolist()
    
    # Save parameters to JSON
    weights_data = {
        'coef': coef,
        'intercept': intercept,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'vocabulary': vocab,
        'idf': idf,
        'ngram_range': tfidf.ngram_range,
        'lowercase': tfidf.lowercase
    }
    
    output_path = os.path.join(model_dir, 'model_weights.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(weights_data, f)
        
    print(f"Successfully saved raw model weights to {output_path}!")

if __name__ == '__main__':
    extract_and_save()
