"""Test out-of-band CF labeling mode with multiclass CF-aware loss."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from utils_v8 import ProcessedDataset, define_models, get_modified_loss_fn

class TimingCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print(f"    [FIT] Training starting...")
        self.train_start = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"    [FIT] Epoch {epoch} beginning...")
        self.epoch_start = time.time()
        
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()
        
    def on_train_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start
        loss_val = logs.get('loss', 0) if logs else 0
        print(f"    [FIT] Batch {batch} completed in {batch_time:.3f}s, loss={loss_val:.4f}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        print(f"    [FIT] Epoch {epoch} completed in {epoch_time:.2f}s")
        
    def on_train_end(self, logs=None):
        train_time = time.time() - self.train_start
        print(f"    [FIT] Training completed in {train_time:.2f}s")

def test_multiclass_cf_loss():
    """Test that multiclass CF-aware loss works with -1 labels."""
    print("Testing multiclass CF-aware loss function...")
    start_overall = time.time()
    
    # Create a small synthetic multiclass dataset
    np.random.seed(42)
    X = np.random.randn(40, 4).astype('float32')
    y = np.random.randint(0, 3, 40).astype('float32')  # 3 classes: 0, 1, 2

    # Add some CF samples with label -1 (smaller)
    X_cf = np.random.randn(8, 4).astype('float32')
    y_cf = np.full(8, -1.0, dtype='float32')  # CF marker

    X_combined = np.vstack([X, X_cf])
    y_combined = np.concatenate([y, y_cf])
    
    # Build a simple multiclass model
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    # Test the multiclass CF-aware loss
    base_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    cf_loss_fn = get_modified_loss_fn(base_loss, k=0.5, loss_type='onesidemod', num_classes=3)
    
    model.compile(
        loss=cf_loss_fn,
        optimizer='adam',
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    
    print("  Model compiled with multiclass CF-aware loss")
    
    # Train for a few epochs to ensure loss computation works
    try:
        print("[DEBUG] Starting model.fit()...")
        start_time = time.time()
        history = model.fit(X_combined, y_combined, epochs=1, batch_size=32)
        elapsed = time.time() - start_time
        print(f"[DEBUG] model.fit() took {elapsed:.2f} seconds")
        print(f"Training successful! Final loss: {history.history['loss'][-1]:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Test prediction on CF samples
    preds_cf = model.predict(X_cf)
    print(f"  Predictions on CF samples shape: {preds_cf.shape}")
    print(f"  Sample CF prediction (first): {preds_cf[0]}")
    
    elapsed_overall = time.time() - start_overall
    print(f"[DEBUG] Total test time: {elapsed_overall:.2f} seconds")
    print("\nMulticlass CF-aware loss test passed!")

def test_out_of_band_cf_label():
    """Test out-of-band cf_label mode - progressive size testing to find bottleneck."""
    import time
    print("\nTesting out-of-band cf_label with PROGRESSIVE SIZE TESTING...")
    print("="*60)
    
    # ====================================================================
    # STEP 1: Test with SYNTHETIC data (fast)
    # ====================================================================
    print("\n[STEP 1] Testing with pure synthetic data (should be FAST)...")
    start = time.time()
    
    np.random.seed(42)
    X_synth = np.random.randn(20, 4).astype('float32')
    y_synth = np.random.randint(0, 3, 20).astype('float32')
    X_cf_synth = np.random.randn(4, 4).astype('float32')
    y_cf_synth = np.full(4, -1.0, dtype='float32')
    
    X_combined = np.vstack([X_synth, X_cf_synth])
    y_combined = np.concatenate([y_synth, y_cf_synth])
    
    # Build simple model manually (not using define_models)
    model_synth = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        keras.layers.Dense(5, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    base_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    cf_loss = get_modified_loss_fn(base_loss, k=0.5, loss_type='onesidemod', num_classes=3)
    model_synth.compile(loss=cf_loss, optimizer='adam')
    
    print(f"  Data ready ({len(X_combined)} samples). Fitting...")
    
    # Add callback to see what's happening during fit    
    fit_start = time.time()
    model_synth.fit(X_combined, y_combined, epochs=1, batch_size=16, callbacks=[TimingCallback()])
    fit_time = time.time() - fit_start
    step1_time = time.time() - start
    print(f"  STEP 1 completed: fit={fit_time:.2f}s, total={step1_time:.2f}s")
    
    # ====================================================================
    # STEP 2: Load Iris dataset
    # ====================================================================
    print("\n[STEP 2] Loading Iris dataset...")
    start = time.time()
    dataset_obj = ProcessedDataset('iris')
    load_time = time.time() - start
    print(f"  Dataset loaded in {load_time:.2f}s")
    
    start = time.time()
    x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()
    split_time = time.time() - start
    print(f"  get_splits() took {split_time:.2f}s")
    print(f"  x_trn shape: {x_trn.shape}, type: {type(x_trn)}")
    
    num_classes = 3
    
    # ====================================================================
    # STEP 3: Train on Iris data without define_models (manual model)
    # ====================================================================
    print("\n[STEP 3] Training on Iris data with MANUAL model (no define_models)...")
    start = time.time()
    
    input_dim = x_trn.shape[1]
    model_manual = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(5, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    cf_loss2 = get_modified_loss_fn(base_loss, k=0.5, loss_type='onesidemod', num_classes=3)
    model_manual.compile(loss=cf_loss2, optimizer='adam')
    
    X_train = x_trn.values[:20]
    y_train = y_trn.values[:20]
    X_cf = x_trn.values[20:24]
    y_cf = np.full(4, -1.0, dtype='float32')
    X_iris = np.vstack([X_train, X_cf])
    y_iris = np.concatenate([y_train, y_cf])
    
    print(f"  Data ready ({len(X_iris)} samples). Fitting...")
    
    fit_start = time.time()
    model_manual.fit(X_iris, y_iris, epochs=1, batch_size=16,callbacks=[TimingCallback()])
    fit_time = time.time() - fit_start
    step3_time = time.time() - start
    print(f"  STEP 3 completed: fit={fit_time:.2f}s, total={step3_time:.2f}s")
    
    # ====================================================================
    # STEP 4: Use define_models
    # ====================================================================
    print("\n[STEP 4] Training with define_models() function...")
    start = time.time()
    
    print(f"  Calling define_models with x_trn shape={x_trn.shape}...")
    define_start = time.time()
    targ_model, surr_models = define_models(x_trn, [10, 5], [[10, 5]], num_classes=num_classes)
    define_time = time.time() - define_start
    print(f"  define_models() took {define_time:.2f}s")
    print(f"  Surrogate input shape: {surr_models[0].input_shape}")
    
    print(f"  Compiling...")
    compile_start = time.time()
    cf_loss3 = get_modified_loss_fn(base_loss, k=0.5, loss_type='onesidemod', num_classes=3)
    surr_models[0].compile(loss=cf_loss3, optimizer='adam')
    compile_time = time.time() - compile_start
    print(f"  Compilation took {compile_time:.2f}s")
    
    print(f"  Fitting on {len(X_iris)} samples...")

    fit_start = time.time()
    surr_models[0].fit(X_iris, y_iris, epochs=1, batch_size=16, verbose=0, callbacks=[TimingCallback()])
    fit_time = time.time() - fit_start
    step4_time = time.time() - start
    print(f"  STEP 4 completed: fit={fit_time:.2f}s, total={step4_time:.2f}s")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "="*60)
    print("PROGRESSIVE TEST SUMMARY:")
    print(f"  Step 1 (synthetic+manual model): {step1_time:.2f}s")
    print(f"  Step 2 (load Iris): {load_time+split_time:.2f}s")
    print(f"  Step 3 (Iris+manual model): {step3_time:.2f}s")
    print(f"  Step 4 (Iris+define_models): {step4_time:.2f}s")
    print("="*60)
    
    if step4_time > 5:
        print("\nSTEP 4 is slow! Problem is likely in define_models() or its usage")
    elif step3_time > 5:
        print("\nSTEP 3 is slow! Problem is with Iris data itself")
    elif load_time + split_time > 5:
        print("\nDataset loading is slow!")
    else:
        print("\n✓ All steps completed quickly!")
    
    print("\nOut-of-band cf_label test passed!")

if __name__ == '__main__':
    test_multiclass_cf_loss()
    test_out_of_band_cf_label()
    print("\n" + "="*60)
    print("All out-of-band mode tests passed!")
    print("="*60)