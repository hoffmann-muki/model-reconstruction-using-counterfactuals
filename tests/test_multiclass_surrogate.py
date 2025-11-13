# test_multiclass_surrogate.py
import numpy as np
import pandas as pd
from utils_v8 import ProcessedDataset, define_models, compile_models, train_models, evaluate_models, generate_query_data
import tensorflow as tf
from tensorflow import keras

print("=== Multiclass Surrogate Model Test (Iris 3-class) ===")

# Load Iris dataset (3 classes)
dataset_obj = ProcessedDataset('iris')
x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()

print(f"Training set: {x_trn.shape}, Test set: {x_tst.shape}")
print(f"Class distribution - Train: {np.bincount(y_trn.astype(int))}")
print(f"Classes in dataset: {np.unique(y_trn)}")

# Create target model for 3-class classification
targ_model, surr_models = define_models(x_trn, targ_arch=[20, 10], surr_archs=[[10], [16]], num_classes=3)

# Compile target model for multiclass
compile_models([targ_model], 
               losses=[keras.losses.SparseCategoricalCrossentropy()],
               optimizers=[keras.optimizers.Adam(0.01)],
               metrics=[[keras.metrics.SparseCategoricalAccuracy()]])

# Train target model
print("Training target model...")
train_models([targ_model], x_trn, y_trn, epochs=30, verbose=1)

# Evaluate target model
target_acc, _ = evaluate_models([targ_model], x_tst, y_tst, num_classes=3)
print(f"Target model accuracy: {target_acc[0]:.4f}")

# Test predictions on a few samples
print("\nTarget model predictions on test samples:")
test_preds = targ_model.predict(x_tst[:5])
test_pred_classes = np.argmax(test_preds, axis=1)
print(f"True labels: {y_tst[:5].values}")
print(f"Predicted classes: {test_pred_classes}")
print(f"Prediction probabilities:\n{test_preds}")

# Generate multiclass counterfactual queries
print("Generating multiclass counterfactual queries...")
exp_dir = generate_query_data('results/test_multiclass_surrogate', 'iris', True, 8, 'naivedat', 'onesided', 'knn', 2, 'TF', 'random',
                              1, 1, [20, 10], 8, 0.01, [[10], [16]], 12, 0.01, [-1, -1], [-1, -1], 32, 
                              cf_target_class=2, num_classes=3)

print(f"Query data generated in: {exp_dir}")

# Load and analyze generated query data
import os
query_file = os.path.join(exp_dir, 'query_000_000.csv')
if os.path.exists(query_file):
    query_data = pd.read_csv(query_file, index_col=0)
    print(f"Generated {len(query_data)} query samples")
    print(f"Query labels distribution: {query_data[targetcol].value_counts().sort_index()}")
    
    # Load pre-trained surrogate models
    surr_model_1 = keras.models.load_model(os.path.join(exp_dir, 'naive_model_00.keras'))
    surr_model_2 = keras.models.load_model(os.path.join(exp_dir, 'naive_model_01.keras'))
    
    # Evaluate surrogate fidelity for multiclass
    surr_models_loaded = [surr_model_1, surr_model_2]
    surr_accs, surr_fids = evaluate_models(surr_models_loaded, x_tst, y_tst, targ_model, num_classes=3)
    
    print("\nMulticlass Surrogate Model Results:")
    for i, (acc, fid) in enumerate(zip(surr_accs, surr_fids)):
        print(f"Surrogate {i+1} - Accuracy: {acc:.4f}, Fidelity: {fid:.4f}")
        
    # Test surrogate predictions vs target predictions
    print("\nComparing surrogate vs target predictions:")
    target_test_preds = np.argmax(targ_model.predict(x_tst[:10]), axis=1)
    surr1_test_preds = np.argmax(surr_model_1.predict(x_tst[:10]), axis=1)
    surr2_test_preds = np.argmax(surr_model_2.predict(x_tst[:10]), axis=1)
    
    print(f"True labels:  {y_tst[:10].values}")
    print(f"Target preds: {target_test_preds}")
    print(f"Surr1 preds:  {surr1_test_preds}")
    print(f"Surr2 preds:  {surr2_test_preds}")
    
    # Calculate per-sample agreement
    agreement_1 = np.mean(target_test_preds == surr1_test_preds)
    agreement_2 = np.mean(target_test_preds == surr2_test_preds)
    print(f"Sample-wise agreement - Surr1: {agreement_1:.4f}, Surr2: {agreement_2:.4f}")
    
else:
    print("Query file not found - experiment may have failed")