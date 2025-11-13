# test_mnist_surrogate.py
import numpy as np
import pandas as pd
from utils_v8 import generate_query_data
import tensorflow as tf
from tensorflow import keras
import os

print("=== MNIST Subsampled Surrogate Test (10-class) ===")

# Generate MNIST experiment with limited samples
print("Running MNIST experiment with sample_limit=300...")
exp_dir = generate_query_data('results/test_mnist_surrogate', 'mnist', True, 6, 'naivedat', 'onesided', 'knn', 2, 'TF', 'random',
                              1, 1, [32, 16], 5, 0.01, [[16], [20]], 8, 0.01, [-1, -1], [-1, -1], 32, 
                              cf_target_class=1, num_classes=10, sample_limit=300)

print(f"MNIST experiment completed in: {exp_dir}")

# Check if experiment generated files
info_file = os.path.join(exp_dir, 'info.csv')
query_file = os.path.join(exp_dir, 'query_000_000.csv')

if os.path.exists(info_file) and os.path.exists(query_file):
    # Load experiment info
    info_df = pd.read_csv(info_file, index_col=0)
    query_data = pd.read_csv(query_file, index_col=0)
    
    print(f"Experiment info: {dict(info_df.iloc[0])}")
    print(f"Generated {len(query_data)} query samples")
    
    # Load target and surrogate models
    target_model = keras.models.load_model(os.path.join(exp_dir, 'targ_model_000.keras'))
    surr_model_1 = keras.models.load_model(os.path.join(exp_dir, 'naive_model_00.keras'))
    surr_model_2 = keras.models.load_model(os.path.join(exp_dir, 'naive_model_01.keras'))
    
    print("\nModel architectures:")
    print(f"Target model: {[layer.units for layer in target_model.layers if hasattr(layer, 'units')]}")
    print(f"Surrogate 1: {[layer.units for layer in surr_model_1.layers if hasattr(layer, 'units')]}")
    print(f"Surrogate 2: {[layer.units for layer in surr_model_2.layers if hasattr(layer, 'units')]}")
    
    # Test on a small subset of query data
    test_X = query_data.drop('label', axis=1).values[:20]
    target_preds = np.argmax(target_model.predict(test_X), axis=1)
    surr1_preds = np.argmax(surr_model_1.predict(test_X), axis=1)
    surr2_preds = np.argmax(surr_model_2.predict(test_X), axis=1)
    
    print(f"\nPrediction comparison (first 20 samples):")
    print(f"Target:     {target_preds}")
    print(f"Surrogate1: {surr1_preds}")
    print(f"Surrogate2: {surr2_preds}")
    
    fidelity_1 = np.mean(target_preds == surr1_preds)
    fidelity_2 = np.mean(target_preds == surr2_preds)
    
    print(f"\nFidelity on sample: Surr1={fidelity_1:.4f}, Surr2={fidelity_2:.4f}")
    
else:
    print("Experiment files missing - check for errors in previous output")