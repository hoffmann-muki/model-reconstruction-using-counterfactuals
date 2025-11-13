# test_binary_surrogate.py
import numpy as np
import pandas as pd
import os
from utils_v8 import ProcessedDataset, define_models, compile_models, train_models, evaluate_models, generate_query_data
import tensorflow as tf
from tensorflow import keras

print("=== Binary Surrogate Model Test (Iris as 2-class) ===")

# Load Iris and convert to binary classification (class 0 vs others)
dataset_obj = ProcessedDataset('iris')
x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()

# Convert to binary: class 0 vs class 1+2
y_trn_binary = (y_trn >= 1).astype('float32')
y_tst_binary = (y_tst >= 1).astype('float32')
y_atk_binary = (y_atk >= 1).astype('float32')

print(f"Training set: {x_trn.shape}, Test set: {x_tst.shape}")
print(f"Binary class distribution - Train: {np.bincount(y_trn_binary.astype(int))}")

# Create target model
targ_model, surr_models = define_models(x_trn, targ_arch=[16, 8], surr_archs=[[8], [12]], num_classes=2)

# Compile target model
compile_models([targ_model], 
               losses=[keras.losses.BinaryCrossentropy()],
               optimizers=[keras.optimizers.Adam(0.01)],
               metrics=[[keras.metrics.BinaryAccuracy()]])

# Train target model
print("Training target model...")
train_models([targ_model], x_trn, y_trn_binary, epochs=20, verbose=1)

# Evaluate target model
target_acc, _ = evaluate_models([targ_model], x_tst, y_tst_binary, num_classes=2)
print(f"Target model accuracy: {target_acc[0]:.4f}")

# Generate some query data using the target model
print("Generating counterfactual queries...")
exp_dir = generate_query_data('results/test_binary_surrogate', 'iris', True, 10, 'naivedat', 'onesided', 'knn', 2, 'TF', 'random',
                              1, 1, [16, 8], 5, 0.01, [[8], [12]], 10, 0.01, [-1, -1], [-1, -1], 32, cf_target_class=1, num_classes=2)

print(f"Query data generated in: {exp_dir}")

# Load generated query data
query_file = os.path.join(exp_dir, 'query_000_000.csv')
if os.path.exists(query_file):
    query_data = pd.read_csv(query_file, index_col=0)
    print(f"Generated {len(query_data)} query samples")
    
    # Use query data to train surrogate models
    query_x = query_data.drop(targetcol, axis=1)
    query_y = query_data[targetcol]
    
    # Load pre-trained surrogate models from experiment
    surr_model_1 = keras.models.load_model(os.path.join(exp_dir, 'naive_model_00.keras'))
    surr_model_2 = keras.models.load_model(os.path.join(exp_dir, 'naive_model_01.keras'))
    
    # Evaluate surrogate fidelity
    surr_models_loaded = [surr_model_1, surr_model_2]
    surr_accs, surr_fids = evaluate_models(surr_models_loaded, x_tst, y_tst_binary, targ_model, num_classes=2)
    
    print("\nSurrogate Model Results:")
    for i, (acc, fid) in enumerate(zip(surr_accs, surr_fids)):
        print(f"Surrogate {i+1} - Accuracy: {acc:.4f}, Fidelity: {fid:.4f}")
else:
    print("Query file not found - experiment may have failed")