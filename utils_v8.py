import sys
sys.path.insert(0, './consistency/')
from consistency import IterativeSearch
from recourse_methods import *
from recourse_utils import *

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

import dice_ml
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets as sk_datasets

import tensorflow as tf
import keras

import datasets
from datasets import load_dataset
from typing import Optional

def generate_test_models(surr_models):
    naive_surr_models = []
    smart_surr_models = []

    for model in surr_models:
        naive_surr_models.append(keras.models.clone_model(model))
        smart_surr_models.append(keras.models.clone_model(model))
    
    return naive_surr_models, smart_surr_models

def generate_bin_models(models):
    def bin_func(x):
        greater = keras.backend.greater_equal(x[:,0], 0.5) # will return boolean values
        greater = keras.backend.cast(greater, dtype=keras.backend.floatx()) # will convert bool to 0 and 1    
        return greater

    return [keras.Model(inputs=model.input, outputs=keras.layers.Lambda(bin_func)(model.output)) \
            for model in models]

def generate_duo_models(models):
    def duo_func(x):
        m = tf.math.subtract(1.0, x) 
        n = tf.concat([x, m], axis=1)
        return n
    return [keras.Model(inputs=model.input, outputs=keras.layers.Lambda(duo_func)(model.output)) for model in models]

def compile_models(models, losses, optimizers, metrics):
    for i in range(len(models)):
        # Keras expects the `metrics` argument to be a list, tuple, or dict.
        # Allow callers to pass a single metric object per-model and wrap it into a list.
        metric_arg = metrics[i]
        if not isinstance(metric_arg, (list, tuple, dict)):
            metric_arg = [metric_arg]
        models[i].compile(loss=losses[i], optimizer=optimizers[i], metrics=metric_arg)

def train_models(models, x_trn, y_trn, epochs, batch_size=32, verbose=0):
    history = []
    for model in models:
        history.append(model.fit(x_trn, y_trn, epochs=epochs, batch_size=batch_size, verbose=verbose))
    return history

def evaluate_models(surr_models, x_ref, y_ref=None, targ_model=None, num_classes: int = 2):
    """Evaluate surrogate models.
    For binary (num_classes==2) keep previous behaviour (threshold 0.5).
    For multiclass use argmax-based accuracy and fidelity.
    Returns: (accuracies_list, fidelities_list)
    """
    accuracies = []
    fidelities = []

    # prepare true labels
    if y_ref is not None:
        # y_ref may be a pandas Series or numpy array of class labels
        y_vals = np.array(y_ref)
        if num_classes == 2:
            # convert to binary 0/1 floats for Keras evaluate
            try:
                # if y_ref is pandas Series with boolean-like values
                y = (y_ref >= 0.5).replace({True: 1.0, False: 0.0})
            except Exception:
                y = (y_vals >= 0.5).astype('float32')
        else:
            # multiclass labels: ensure integer dtype for sparse evaluation
            y = y_vals.astype('int32')
    else:
        y = None

    # prepare target predictions for fidelity
    if targ_model is not None:
        u_pred = targ_model.predict(x_ref)
        if num_classes == 2:
            u = (u_pred >= 0.5).astype('int32')
        else:
            u = np.argmax(u_pred, axis=1)
    else:
        u = None

    for surr_model in surr_models:
        # fidelity: compare surrogate predictions to target predictions
        if targ_model is not None:
            v_pred = surr_model.predict(x_ref)
            if num_classes == 2:
                v = (v_pred >= 0.5).astype('int32')
                mismatch = np.where(u != v, 1, 0)
            else:
                v = np.argmax(v_pred, axis=1)
                mismatch = np.where(u != v, 1, 0)
            fidelity = 1 - np.sum(mismatch) / len(x_ref)
            fidelities.append(fidelity)

        # accuracy: compare surrogate predictions to ground-truth labels
        if y is not None:
            if num_classes == 2:
                # Keras evaluate returns [loss, metric], we use model.evaluate to get metric
                try:
                    acc = surr_model.evaluate(x_ref, y, verbose=0)[1]
                except Exception:
                    # fallback to numpy-based accuracy
                    preds = (surr_model.predict(x_ref) >= 0.5).astype('int32').reshape(-1)
                    acc = float(np.mean(preds == np.array(y).astype('int32')))
                accuracies.append(acc)
            else:
                preds = np.argmax(surr_model.predict(x_ref), axis=1)
                acc = float(np.mean(preds == np.array(y).astype('int32')))
                accuracies.append(acc)

    return accuracies, fidelities

def reset_weights(models, seed=None):
    for model in models:
        for layer in model.layers:         
            kernel_init = keras.initializers.RandomUniform(maxval=1, minval=-1, seed=seed)
            bias_init = keras.initializers.Zeros()
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                layer.set_weights([kernel_init(shape=np.asarray(layer.kernel.shape)), \
                               bias_init(shape=np.asarray(layer.bias.shape))])
                
# class ProcessedDataset:
#     def __init__(self, dataset):
#         datasets.utils.logging.set_verbosity(datasets.logging.ERROR)
#         if dataset == 'adultincome':
#             raw_data = load_dataset("jlh/uci-adult-income")["train"]
#             raw_data = pd.DataFrame(raw_data)
#             raw_data_full = pd.DataFrame(raw_data)

#             targetcol = "income"
#             g = raw_data.groupby(targetcol)
#             raw_data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))

#             # drop some of the negative samples
#             raw_data_full = raw_data_full.drop(raw_data_full[raw_data_full[targetcol] < 1].sample(frac=0.9).index)

#             numcols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
#             catcols = raw_data_full.columns.difference([*numcols, targetcol])
#             dataframe = raw_data.reindex(columns=[*numcols, *catcols, targetcol])
#             dataframe_full = raw_data_full.reindex(columns=[*numcols, *catcols, targetcol])
#             for catcol in catcols:
#                 categories = dataframe[catcol].unique()
#                 numerals = [float(i) for i in range(len(categories))]
#                 dataframe[catcol] = dataframe[catcol].replace(categories, numerals)
#                 dataframe_full[catcol] = dataframe_full[catcol].replace(categories, numerals)

#         elif dataset == 'heloc':
#             raw_data = load_dataset("mstz/heloc")["train"]
#             raw_data = pd.DataFrame(raw_data)
#             targetcol = "is_at_risk"
#             numcols = ['estimate_of_risk', 'net_fraction_of_revolving_burden', 'percentage_of_legal_trades',
#                         'months_since_last_inquiry_not_recent', 'months_since_last_trade', 'percentage_trades_with_balance', 
#                         'number_of_satisfactory_trades', 'average_duration_of_resolution', 'nr_total_trades', 
#                         'nr_banks_with_high_ratio']
#             catcols = []
#             raw_data = raw_data.drop(raw_data.columns.difference([*numcols, targetcol]), axis=1)
#             dataframe = raw_data
#             raw_data_full = raw_data
#             dataframe_full = dataframe

#         elif dataset == 'compas':
#             train_data, test_data = load_dataset("imodels/compas-recidivism", split =['train', 'test'])
#             raw_data = pd.concat([pd.DataFrame(train_data), pd.DataFrame(test_data)], axis=0)
#             targetcol = "is_recid"
#             numcols = list(raw_data.columns.difference([targetcol]))
#             catcols = []
#             dataframe = raw_data
#             raw_data_full = raw_data
#             dataframe_full = dataframe

#         elif dataset == 'defaultcredit':
#             raw_data = load_dataset("scikit-learn/credit-card-clients")["train"]
#             raw_data = pd.DataFrame(raw_data)
#             targetcol = "default.payment.next.month"
#             print(raw_data.info())
#             print(raw_data[targetcol].value_counts())
#             g = raw_data.groupby(targetcol)
#             raw_data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
#             numcols = list(raw_data.columns.difference([targetcol]))
#             catcols = []
#             dataframe = raw_data
#             raw_data_full = raw_data
#             dataframe_full = dataframe


#         dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
#         dataframe[targetcol] = raw_data[targetcol]

#         dataframe_full = (dataframe_full-dataframe_full.min())/(dataframe_full.max()-dataframe_full.min())
#         dataframe_full[targetcol] = raw_data_full[targetcol]

#         self.dataframe = dataframe
#         self.dataframe_full = dataframe_full
#         self.targetcol = targetcol
#         self.numcols = numcols
#         self.catcols = catcols

#     def get_splits(self, shuffle=True):
#         # prepare datasets - train (50%), test (25%), attack (25%)
#         target = self.dataframe[self.targetcol]
#         trn_df, tst_df, y_trn, y_tst = sklearn.model_selection.train_test_split(self.dataframe,
#                                                                         target,
#                                                                         test_size=0.5,
#                                                                         random_state=0,
#                                                                         shuffle=shuffle,
#                                                                         stratify=target)

#         atk_df, tst_df, y_atk, y_tst = sklearn.model_selection.train_test_split(tst_df,
#                                                                         y_tst,
#                                                                         test_size=0.5,
#                                                                         random_state=0,
#                                                                         shuffle=shuffle,
#                                                                         stratify=y_tst)

#         x_trn = trn_df.drop(self.targetcol, axis=1)
#         x_tst = tst_df.drop(self.targetcol, axis=1)
#         x_atk = atk_df.drop(self.targetcol, axis=1)
#         y_trn = y_trn.astype('float32')
#         y_tst = y_tst.astype('float32')
#         y_atk = y_atk.astype('float32')

#         dfs = [self.dataframe, self.dataframe_full]
#         return [x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dfs, self.numcols, self.catcols, self.targetcol]

class ProcessedDataset:
    def __init__(self, dataset, sample_limit: Optional[int] = None):
        datasets.utils.logging.set_verbosity(datasets.logging.ERROR)
        if dataset == 'adultincome':
            raw_data = load_dataset("jlh/uci-adult-income")["train"]
            raw_data = pd.DataFrame(raw_data)
            targetcol = "income"
            
            numcols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
            catcols = raw_data.columns.difference([*numcols, targetcol])
            dataframe = raw_data.reindex(columns=[*numcols, *catcols, targetcol])

            for catcol in catcols:
                categories = dataframe[catcol].unique()
                numerals = [float(i) for i in range(len(categories))]
                dataframe[catcol] = dataframe[catcol].replace(categories, numerals)
    
        elif dataset == 'heloc':
            raw_data = load_dataset("mstz/heloc")["train"]
            raw_data = pd.DataFrame(raw_data)
            targetcol = "is_at_risk"
            numcols = ['estimate_of_risk', 'net_fraction_of_revolving_burden', 'percentage_of_legal_trades',
                        'months_since_last_inquiry_not_recent', 'months_since_last_trade', 'percentage_trades_with_balance', 
                        'number_of_satisfactory_trades', 'average_duration_of_resolution', 'nr_total_trades', 
                        'nr_banks_with_high_ratio']
            catcols = []
            raw_data = raw_data.drop(raw_data.columns.difference([*numcols, targetcol]), axis=1)
            dataframe = raw_data

        elif dataset == 'compas':
            train_data, test_data = load_dataset("imodels/compas-recidivism", split =['train', 'test'])
            raw_data = pd.concat([pd.DataFrame(train_data), pd.DataFrame(test_data)], axis=0)
            targetcol = "is_recid"
            numcols = list(raw_data.columns.difference([targetcol]))
            catcols = []
            dataframe = raw_data
            
        elif dataset == 'defaultcredit' or dataset == 'dccc':
            raw_data = load_dataset("scikit-learn/credit-card-clients")["train"]
            raw_data = pd.DataFrame(raw_data)
            targetcol = "default.payment.next.month"
            numcols = list(raw_data.columns.difference([targetcol]))
            catcols = []
            dataframe = raw_data
        
        elif dataset == 'iris':
            # small multiclass tabular dataset
            # load_iris can return either a Bunch or (data, target); request a tuple for consistent indexing
            X, y = sk_datasets.load_iris(return_X_y=True)
            targetcol = 'label'
            df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
            df[targetcol] = y
            dataframe = df
            numcols = [c for c in df.columns if c != targetcol]
            catcols = []

        elif dataset == 'mnist':
            # handwritten digits (multiclass) - flatten images
            (x_trn, y_trn), (x_tst, y_tst) = keras.datasets.mnist.load_data()
            X = np.concatenate([x_trn, x_tst], axis=0)
            y = np.concatenate([y_trn, y_tst], axis=0)
            # subsample if requested
            if sample_limit is not None and sample_limit < X.shape[0]:
                idx = np.random.choice(X.shape[0], sample_limit, replace=False)
                X = X[idx]
                y = y[idx]
            X = X.reshape((X.shape[0], -1)).astype('float32') / 255.0
            targetcol = 'label'
            df = pd.DataFrame(X)
            df[targetcol] = y
            dataframe = df
            numcols = [c for c in df.columns if c != targetcol]
            catcols = []

        elif dataset == 'cifar':
            # CIFAR-10 (multiclass) - flatten RGB images
            (x_trn, y_trn), (x_tst, y_tst) = keras.datasets.cifar10.load_data()
            X = np.concatenate([x_trn, x_tst], axis=0)
            y = np.concatenate([y_trn, y_tst], axis=0).squeeze()
            # subsample if requested
            if sample_limit is not None and sample_limit < X.shape[0]:
                idx = np.random.choice(X.shape[0], sample_limit, replace=False)
                X = X[idx]
                y = y[idx]
            X = X.reshape((X.shape[0], -1)).astype('float32') / 255.0
            targetcol = 'label'
            df = pd.DataFrame(X)
            df[targetcol] = y
            dataframe = df
            numcols = [c for c in df.columns if c != targetcol]
            catcols = []
        
        # normalize data (features to [0,1])
        # Restore target column from available label sources when necessary
        raw_target = None
        raw_data_local = locals().get('raw_data', None)
        if raw_data_local is not None:
            try:
                # ensure targetcol is defined in this scope before attempting to index raw_data_local
                if 'targetcol' in locals():
                    raw_target = raw_data_local[targetcol] # type: ignore
                else:
                    raw_target = None
            except Exception:
                raw_target = None

        dataframe = (dataframe - dataframe.min())/(dataframe.max()-dataframe.min())
        if raw_target is not None:
            dataframe[targetcol] = raw_target
        elif 'y' in locals():
            dataframe[targetcol] = pd.Series(y, index=dataframe.index)
        elif 'y_trn' in locals() and 'y_tst' in locals():
            y_all = np.concatenate([y_trn, y_tst], axis=0)
            dataframe[targetcol] = pd.Series(y_all, index=dataframe.index)
        else:
            # fallback: assume target column already present
            dataframe[targetcol] = dataframe[targetcol]

        self.dataframe = dataframe
        self.targetcol = targetcol
        self.numcols = numcols
        self.catcols = catcols

        self.class_value_counts = self.dataframe[targetcol].value_counts()

    def split_dataframe(self, df, class_column, train_size, test_size, attack_sizes):
        # Initialize DataFrames for train, test, and remaining
        train_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)
        attack_df = pd.DataFrame(columns=df.columns)
        remaining_df = df.copy()  # Start with all data in remaining

        # Get unique classes
        classes = remaining_df[class_column].unique()

        for class_label in classes:
            # Get all samples for the current class
            class_df = remaining_df[remaining_df[class_column] == class_label]

            # Sample for training
            train_samples_extracted = class_df.sample(n=train_size, random_state=42)
            
            # Sample for testing from the remaining class samples
            class_df = class_df.drop(train_samples_extracted.index)
            test_samples_extracted = class_df.sample(n=test_size, random_state=42)

            # Sample for attack from the remaining class samples
            class_df = class_df.drop(test_samples_extracted.index)
            attack_samples_extracted = class_df.sample(n=attack_sizes[class_label], random_state=42)
            
            # Update DataFrames
            train_df = pd.concat([train_df, train_samples_extracted], ignore_index=True)
            test_df = pd.concat([test_df, test_samples_extracted], ignore_index=True)
            attack_df = pd.concat([attack_df, attack_samples_extracted], ignore_index=True)

            # Update the remaining DataFrame
            remaining_df = remaining_df.drop(train_samples_extracted.index)
            remaining_df = remaining_df.drop(test_samples_extracted.index)
            remaining_df = remaining_df.drop(attack_samples_extracted.index)

        return train_df, test_df, attack_df

    def get_splits(self, attack_balance=0.5):
        # prepare datasets - train (50%), test (25%), attack (remaining, with a ratio class0/total=attack_balance)
        balanced_size_per_class = self.class_value_counts.min()
        
        test_size = int(np.round(0.25 * balanced_size_per_class))
        train_size = int(np.round(0.5 * balanced_size_per_class))
        remaining_size = balanced_size_per_class - test_size - train_size

        # Build attack_sizes per class. For binary, preserve previous attack_balance behavior.
        classes = self.dataframe[self.targetcol].unique()
        if len(classes) == 2:
            if attack_balance < 0.5:
                attack_sizes = {
                    classes[0]: int(np.round(remaining_size * attack_balance / (1-attack_balance))),
                    classes[1]: remaining_size
                }
            else:
                attack_sizes = {
                    classes[0]: remaining_size,
                    classes[1]: int(np.round(remaining_size * (1-attack_balance) / attack_balance))
                }
        else:
            # For multiclass, allocate the remaining_size equally to each class by default
            attack_sizes = {cls: remaining_size for cls in classes}

        # print('attack sizes:', attack_sizes,
        #       'total samples:', 2*(test_size+train_size) + attack_sizes[0] + attack_sizes[1],
        #       'actual total size:', len(self.dataframe))

        trn_df, tst_df, atk_df = self.split_dataframe(self.dataframe,
                                                    self.targetcol,
                                                    train_size,
                                                    test_size,
                                                    attack_sizes)

        x_trn = trn_df.drop(self.targetcol, axis=1)
        x_tst = tst_df.drop(self.targetcol, axis=1)
        x_atk = atk_df.drop(self.targetcol, axis=1)
        y_trn = trn_df[self.targetcol].astype('float32')
        y_tst = tst_df[self.targetcol].astype('float32')
        y_atk = atk_df[self.targetcol].astype('float32')

        return [x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, self.dataframe, self.numcols, self.catcols, self.targetcol]



def get_multiclass_cf_loss_fn(base_loss, num_classes, k=0.5):
    """
    Simplified multiclass CF-aware loss function.
    - Samples with y_true == -1 are counterfactuals (out-of-band marker)
    - Regular samples (y_true >= 0) use standard SparseCategoricalCrossentropy
    - CF samples get minimal loss (just ignore them or use small entropy penalty)
    
    Args:
        base_loss: SparseCategoricalCrossentropy loss function
        num_classes: number of classes
        k: confidence threshold for CF loss reduction (default 0.5)
    """
    eps = 1e-7
    
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Identify CF samples (y_true == -1)
        is_cf = tf.equal(y_true, -1.0)
        is_regular = tf.logical_not(is_cf)
        
        # For regular samples: use standard sparse categorical crossentropy
        # Replace CF labels (-1) with 0 to avoid errors, but we'll mask them out
        y_true_safe = tf.where(is_regular, y_true, tf.zeros_like(y_true))
        y_true_int = tf.cast(y_true_safe, tf.int32)
        
        # Compute standard cross-entropy loss
        # Use reduction='none' to get per-sample losses for masking
        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_int, y_pred, from_logits=False
        )
        
        # Apply mask: only count regular samples, zero out CF samples
        is_regular_float = tf.cast(is_regular, tf.float32)
        masked_loss = per_sample_loss * is_regular_float
        
        # Average only over regular samples
        n_regular = tf.reduce_sum(is_regular_float)
        regular_loss = tf.reduce_sum(masked_loss) / tf.maximum(n_regular, 1.0)
        
        # For CF samples: use a very small fixed penalty or ignore them
        # (In practice, ignoring CF samples by masking is the simplest approach)
        # If you want CF-aware training, just return regular_loss
        return regular_loss
    
    return loss_fn


def get_modified_loss_fn(base_loss, k, loss_type, num_classes=2):
    """
    Get a modified loss function for binary or multiclass classification with CF-aware training.
    
    Args:
        base_loss: base loss function (BinaryCrossentropy or SparseCategoricalCrossentropy)
        k: CF threshold parameter (or -1 for ordinary loss, -2 for bcecf)
        loss_type: type of CF-aware loss ('ordinary', 'bcecf', 'onesidemod', 'twosidemod')
        num_classes: number of classes (2 for binary, >2 for multiclass)
    
    Returns:
        A loss function compatible with Keras model.compile()
    """
    # For multiclass with cf_label=-1 (out-of-band), use the multiclass CF-aware loss
    # This is detected when num_classes > 2 and loss_type is not 'ordinary'
    if num_classes > 2 and loss_type != 'ordinary':
        return get_multiclass_cf_loss_fn(base_loss, num_classes, k=0.5)
    
    # Binary CF-aware loss (original implementation)
    # Evaluate k as a Python float when possible to avoid Python boolean checks on tensors in graph mode
    try:
        k_py = float(k)
    except Exception:
        k_py = None

    is_k_neg1 = (k_py == -1)
    is_k_neg2 = (k_py == -2)
    k_tensor = tf.constant(k_py if k_py is not None else k, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_true_bin = tf.dtypes.cast(tf.math.greater_equal(y_true, 0.5), tf.float32)

        # Control flow based on Python-evaluated flags where possible
        if is_k_neg1 or loss_type == 'ordinary':
            loss = base_loss(y_true_bin, y_pred)
        elif is_k_neg2 or loss_type == 'bcecf':
            # y_true_cf = 4 * y_true * (1 - y_true)
            y_true_cf = tf.math.multiply(4.0, tf.math.multiply(y_true, tf.math.subtract(1.0, y_true)))
            yy = tf.math.add(tf.math.multiply(0.5, y_true_cf), y_true_bin)
            loss = base_loss(yy, y_pred)
        elif loss_type == 'twosidemod':
            mask_pos_cf = tf.dtypes.cast(tf.math.equal(y_true, 0.5), tf.float32)
            mask_neg_cf = tf.dtypes.cast(tf.math.equal(y_true, -0.5), tf.float32)
            mask_ordinary = tf.dtypes.cast(tf.math.not_equal(tf.math.square(y_true), 0.25), tf.float32)

            eps = 1e-5
            # build the positive-CF loss term
            pos_mask = tf.dtypes.cast(tf.math.less_equal(y_pred, k_tensor), tf.float32)
            pos_term = tf.math.add(
                tf.math.multiply(k_tensor, tf.math.log(tf.math.divide(tf.math.add(k_tensor, eps), tf.math.add(y_pred, eps)))),
                tf.math.multiply(tf.math.subtract(1.0, k_tensor), tf.math.log(tf.math.divide(tf.math.add(tf.math.subtract(1.0, k_tensor), eps), tf.math.add(tf.math.subtract(1.0, y_pred), eps))))
            )
            cf_pos_loss = tf.math.multiply(tf.math.multiply(tf.dtypes.cast(pos_mask, tf.float32), mask_pos_cf), pos_term)

            # build the negative-CF loss term
            neg_mask = tf.dtypes.cast(tf.math.less_equal(tf.math.subtract(1.0, k_tensor), y_pred), tf.float32)
            neg_term = tf.math.add(
                tf.math.multiply(k_tensor, tf.math.log(tf.math.divide(tf.math.add(k_tensor, eps), tf.math.add(tf.math.subtract(1.0, y_pred), eps)))),
                tf.math.multiply(tf.math.subtract(1.0, k_tensor), tf.math.log(tf.math.divide(tf.math.add(tf.math.subtract(1.0, k_tensor), eps), tf.math.add(y_pred, eps))))
            )
            cf_neg_loss = tf.math.multiply(tf.math.multiply(tf.dtypes.cast(neg_mask, tf.float32), mask_neg_cf), neg_term)

            # ordinary binary cross-entropy part
            bce_part = - (tf.math.multiply(y_true, tf.math.log(tf.math.add(y_pred, eps))) + tf.math.multiply(tf.math.subtract(1.0, y_true), tf.math.log(tf.math.add(tf.math.subtract(1.0, y_pred), eps))))
            ord_loss = tf.math.multiply(mask_ordinary, bce_part)
            loss = tf.reduce_mean(tf.math.add_n([cf_pos_loss, cf_neg_loss, ord_loss]))
        else:
            y_true_cf = tf.dtypes.cast(tf.math.equal(y_true, 0.5), tf.float32)
            y_valid_cf = tf.dtypes.cast(tf.math.greater_equal(y_true, 0.0), tf.float32)

            eps = 1e-5
            mask_pred_le_k = tf.dtypes.cast(tf.math.less_equal(y_pred, k_tensor), tf.float32)
            cf_term = tf.math.add(
                tf.math.multiply(k_tensor, tf.math.log(tf.math.divide(tf.math.add(k_tensor, eps), tf.math.add(y_pred, eps)))),
                tf.math.multiply(tf.math.subtract(1.0, k_tensor), tf.math.log(tf.math.divide(tf.math.add(tf.math.subtract(1.0, k_tensor), eps), tf.math.add(tf.math.subtract(1.0, y_pred), eps))))
            )
            cf_loss = tf.math.multiply(tf.math.multiply(tf.dtypes.cast(mask_pred_le_k, tf.float32), y_true_cf), cf_term)

            bce_loss = tf.math.multiply(tf.math.subtract(1.0, y_true_cf), - (tf.math.multiply(y_true, tf.math.log(tf.math.add(y_pred, eps))) + tf.math.multiply(tf.math.subtract(1.0, y_true), tf.math.log(tf.math.add(tf.math.subtract(1.0, y_pred), eps)))))
            loss = tf.reduce_mean(tf.boolean_mask(tf.math.add(cf_loss, bce_loss), tf.math.greater_equal(y_true, 0.0)))
        return loss

    return loss_fn


class Query_Gen:
    def __init__(self, dataframe, categorical_cols, numerical_cols):
        self.dataframe = dataframe
        
        self.categories = {}
        for catcol in categorical_cols:
            self.categories[catcol] = dataframe[catcol].unique()
        
        self.ranges = {}
        for numcol in numerical_cols:
            self.ranges[numcol] = (dataframe[numcol].min(), dataframe[numcol].max())

    def generate_queries(self, N, method="naiveuni", model=None):
        if method == "naiveuni":
            queries = pd.DataFrame(columns=self.dataframe.columns)
            for catcol in self.categories.keys():
                queries[catcol] = np.random.choice(self.categories[catcol], N)
            for numcol in self.ranges.keys():
                queries[numcol] = np.random.uniform(self.ranges[numcol][0], self.ranges[numcol][1], N)
        elif method == "naivedat":
            queries = self.dataframe.sample(n=N, ignore_index=True)
        elif method == "smartuni":
            # Create data_range from the ranges dictionary
            feature_cols = list(self.categories.keys()) + list(self.ranges.keys())
            data_range = []
            for col in feature_cols:
                if col in self.ranges:
                    data_range.append([self.ranges[col][0], self.ranges[col][1]])
                else:
                    # For categorical columns, use min/max of unique values
                    cat_vals = self.categories[col]
                    data_range.append([min(cat_vals), max(cat_vals)])
            data_range = np.array(data_range)
            n_features = len(feature_cols)
            
            queries = np.empty([0, model.input_shape[1]], np.float32)
            while queries.shape[0] < N:
                naive_queries = np.random.uniform(low=data_range[:,0], high=data_range[:,1], size=(N, n_features))
                y_hat = model.predict(naive_queries)
                positive_indices = np.where(y_hat > 0.5)[0]
                if len(positive_indices) > 0:
                    queries = np.concatenate([queries, naive_queries[positive_indices]], axis=0)
        elif method == "smartdat":
            queries = np.empty([0, model.input_shape[1]], np.float32)
            while queries.shape[0] < N:
                # Use dataframe sampling instead of undefined data_distrib
                naive_queries = self.dataframe.sample(n=N, replace=True).drop(self.dataframe.columns[-1], axis=1)
                y_hat = model.predict(naive_queries)
                positive_indices = np.where(y_hat > 0.5)[0]
                if len(positive_indices) > 0:
                    queries = np.concatenate([queries, naive_queries.iloc[positive_indices].values], axis=0)
        else:
            print("generate_queries(): {} is not a valid query generation method".format(method))
        return queries[:N]
    

class Query_API:
    def __init__(self, model, dataframe, cts_features, out_name, method, generator, norm, 
                 dice_backend, dice_method, dice_posthoc_sparsity_param, 
                 dice_proximity_weight, dice_features_to_vary, knn_k, roar_lambda, roar_delta_max, cf_label,
                 cf_target_class=None, num_classes: int = 2):
        self.model = model
        self.out_name = out_name
        self.method = method
        self.cf_label = cf_label
        self.cf_target_class = cf_target_class
        self.num_classes = num_classes

        if generator == "itersearch" or generator=="mccf": # MCCF counterfactuals
            feature_cols = dataframe.columns[:-1]
            feature_vals = dataframe[feature_cols]
            # ensure IterativeSearch receives multiclass information when available
            try:
                nc = int(num_classes)
            except Exception:
                nc = 2
            L2_iter_search = IterativeSearch(generate_duo_models([model])[0],
                                clamp=[feature_vals.min(), feature_vals.max()],
                                num_classes=nc,
                                eps=0.01,
                                nb_iters=100,
                                eps_iter=0.02,
                                norm=norm,
                                sns_fn=None)
            def generate_counterfactuals(x):
                cf = []
                try:
                    cf, pred_cf, is_valid = L2_iter_search(np.array(x))
                    cf = cf[is_valid]
                except:
                    print('No counterfactuals found')
                cfs_df = pd.DataFrame(cf, columns=feature_cols)
                # cfs_df[out_name] = 0.5
                cfs_df = self.get_labeled_cfs(cfs=cfs_df, out_name=out_name)
                return cfs_df

            self.generate_counterfactuals = generate_counterfactuals

        elif generator == 'knn': # nearest neighbor counterfactuals
            feature_cols = dataframe.columns[:-1]
            feature_vals = dataframe[feature_cols]
            # Build neighbor pools per class for multiclass-aware CFs
            preds = self.model.predict(feature_vals)
            if preds.ndim == 1 or (hasattr(preds, 'shape') and preds.shape[1] == 1):
                # binary: pool of samples with prediction==0 or 1 depending on cf_target_class
                labels = (preds >= 0.5).astype('int32').reshape(-1)
            else:
                labels = np.argmax(preds, axis=1)

            # Build per-class pools and NearestNeighbors models
            pools = {}
            nbr_models = {}
            for cls in np.unique(labels):
                pool_df = feature_vals.loc[labels == cls]
                if len(pool_df) > 0:
                    try:
                        nbr = NearestNeighbors(n_neighbors=min(knn_k, len(pool_df)), algorithm='auto').fit(pool_df)
                        pools[int(cls)] = pool_df
                        nbr_models[int(cls)] = nbr
                    except Exception:
                        pools[int(cls)] = pool_df

            def generate_counterfactuals(x):
                # x: DataFrame of queries (features)
                if len(x) == 0:
                    cfs_df = pd.DataFrame([], columns=dataframe.columns)
                    cfs_df.loc[:, out_name] = 0.5 if self.num_classes == 2 else -1
                    return cfs_df

                # Determine target class for CF: use cf_target_class if provided; else choose a class different from predicted
                raw_preds_q = self.model.predict(x)
                if raw_preds_q.ndim == 1 or (hasattr(raw_preds_q, 'shape') and raw_preds_q.shape[1] == 1):
                    q_labels = (raw_preds_q >= 0.5).astype('int32').reshape(-1)
                else:
                    q_labels = np.argmax(raw_preds_q, axis=1)

                target_classes = []
                for ql in q_labels:
                    if self.cf_target_class is not None:
                        tcls = int(self.cf_target_class)
                    else:
                        # choose any class not equal to current prediction; prefer next class modulo num_classes
                        if self.num_classes > 1:
                            tcls = (int(ql) + 1) % max(2, self.num_classes)
                        else:
                            tcls = 1 - int(ql)
                    target_classes.append(tcls)

                # For each query, get nearest neighbors from the pool of the target class
                # Build a global neighbor model for fallback searches across classes
                cfs_list = []
                try:
                    nbr_all = NearestNeighbors(n_neighbors=min(max(10, knn_k), len(feature_vals)), algorithm='auto').fit(feature_vals)
                except Exception:
                    nbr_all = None

                for idx, q in enumerate(x.values):
                    tcls = target_classes[idx]
                    pool_df = pools.get(tcls, None)
                    nbr = nbr_models.get(tcls, None)
                    if pool_df is None or nbr is None or len(pool_df) == 0:
                        # Try to find nearest neighbor of the desired class using the global index
                        found = False
                        if nbr_all is not None:
                            ksearch = min(50, len(feature_vals))
                            try:
                                dists_all, inds_all = nbr_all.kneighbors(q.reshape(1, -1), n_neighbors=ksearch)
                                for cand_idx in inds_all[0]:
                                    cand_label = int(labels[cand_idx])
                                    if cand_label == tcls:
                                        chosen = feature_vals.iloc[cand_idx]
                                        cfs_list.append(np.concatenate([chosen.values, [tcls]]))
                                        found = True
                                        break
                            except Exception:
                                found = False

                        if found:
                            continue

                        # no sample in desired class nearby; fall back to closest sample of any class
                        if nbr_all is not None:
                            try:
                                dists_all, inds_all = nbr_all.kneighbors(q.reshape(1, -1), n_neighbors=1)
                                chosen = feature_vals.iloc[inds_all[0][0]]
                                fallback_label = int(labels[inds_all[0][0]])
                                cfs_list.append(np.concatenate([chosen.values, [fallback_label]]))
                                continue
                            except Exception:
                                pass

                        # ultimate fallback: zero-vector + target label (or 0.5 for binary)
                        if self.num_classes == 2:
                            cf_row = np.concatenate([np.zeros(feature_vals.shape[1]), [0.5]])
                        else:
                            cf_row = np.concatenate([np.zeros(feature_vals.shape[1]), [tcls]])
                        cfs_list.append(cf_row)
                        continue

                    dists, inds = nbr.kneighbors(q.reshape(1, -1))
                    # take first neighbor
                    chosen = pool_df.iloc[inds[0][0]]
                    cfs_list.append(np.concatenate([chosen.values, [tcls]]))

                cfs_arr = np.array(cfs_list)
                # Construct DataFrame with same columns + target
                cols = list(feature_cols) + [out_name]
                cfs_df = pd.DataFrame(cfs_arr, columns=cols)
                return cfs_df

            self.generate_counterfactuals = generate_counterfactuals

        elif generator == 'roar': # ROAR counterfactuals
            print(f'ROAR delta_max: {roar_delta_max}')
            feature_cols = dataframe.columns[:-1]
            feature_vals = dataframe[feature_cols]
            
            recourses=[]
            deltas=[]
            # prepare global neighbor index and predicted labels for ROAR fallback
            try:
                preds_all = self.model.predict(feature_vals)
                if preds_all.ndim == 1 or (hasattr(preds_all, 'shape') and preds_all.shape[1] == 1):
                    lbls_all = (preds_all >= 0.5).astype('int32').reshape(-1)
                else:
                    lbls_all = np.argmax(preds_all, axis=1)
                try:
                    ro_nbr_all = NearestNeighbors(n_neighbors=min(50, len(feature_vals)), algorithm='auto').fit(feature_vals)
                except Exception:
                    ro_nbr_all = None
            except Exception:
                ro_nbr_all = None
                lbls_all = None
            def generate_counterfactuals_roar(queries):
                recourses=[]
                deltas=[]
                def baseline_model(x):
                    x = pd.DataFrame(x, columns=feature_cols)
                    preds = generate_duo_models([model])[0].predict(x)
                    # ensure multiclass-safe baseline: return integer labels for target selection
                    if preds.ndim == 1 or (hasattr(preds, 'shape') and preds.shape[1] == 1):
                        return (preds >= 0.5).astype('int32')
                    else:
                        return np.argmax(preds, axis=1).reshape(-1, 1)
                
                for xi in tqdm(range(len(queries))):
                    x = queries.iloc[xi]
                    try:
                        bt = baseline_model(np.array(x.values).reshape(1,-1))
                        # interpret baseline output as label
                        if bt.ndim == 1 or (hasattr(bt, 'shape') and bt.shape[1] == 1):
                            y_target = int(bt.reshape(-1)[0])
                        else:
                            y_target = int(bt.reshape(-1)[0])
                        np.random.seed(xi)
                        coefficients, intercept = lime_explanation(baseline_model, feature_vals.values, x.values)
                        robust_recourse = RobustRecourse(W=coefficients, W0=intercept, 
                                                        feature_costs=None, y_target=y_target,
                                                        delta_max=roar_delta_max)
                        r, delta_r = robust_recourse.get_recourse(x.values, lamb=roar_lambda)
                        recourses.append(r)
                        deltas.append(delta_r)
                    except Exception as e:
                        # try to find a real sample of the desired class near the query using the global index
                        found = False
                        try:
                            if self.num_classes > 2 and ro_nbr_all is not None and lbls_all is not None:
                                ksearch = min(50, len(feature_vals))
                                dists_all, inds_all = ro_nbr_all.kneighbors(x.values.reshape(1, -1), n_neighbors=ksearch)
                                for cand_idx in inds_all[0]:
                                    if int(lbls_all[cand_idx]) == y_target:
                                        chosen = feature_vals.iloc[cand_idx]
                                        recourses.append(chosen.values)
                                        deltas.append(0.0)
                                        found = True
                                        break
                        except Exception:
                            found = False

                        if found:
                            continue

                        # try a binary fallback if multiclass recourse fails
                        try:
                            if self.num_classes > 2:
                                # attempt to coerce to binary target (0 or 1) by mapping multiclass target to 0/1
                                fallback_target = 1 if y_target != 1 else 0
                                robust_recourse = RobustRecourse(W=coefficients, W0=intercept, 
                                                                feature_costs=None, y_target=fallback_target,
                                                                delta_max=roar_delta_max)
                                r, delta_r = robust_recourse.get_recourse(x.values, lamb=roar_lambda)
                                recourses.append(r)
                                deltas.append(delta_r)
                                continue
                        except Exception:
                            pass

                        print(f'no counterfactuals generated for query {xi}')
                        print(e)

                cfs_df = pd.DataFrame(recourses, columns=feature_cols)
                cfs_df = self.get_labeled_cfs(cfs=cfs_df, out_name=out_name)
                return cfs_df
            self.generate_counterfactuals = generate_counterfactuals_roar
            
        else: # DiCE counterfactuals
            m = dice_ml.Model(model=model, backend=dice_backend)
            d = dice_ml.Data(dataframe=dataframe, continuous_features=cts_features, outcome_name=out_name)
            e = dice_ml.Dice(d, m, method=dice_method)

            def generate_counterfactuals(x):
                try:
                    # Determine desired_class for DiCE: explicit cf_target_class for multiclass,
                    # 'opposite' for binary, or omit for DiCE default when multiclass target not provided
                    if self.cf_target_class is not None:
                        desired = int(self.cf_target_class)
                        exp = e.generate_counterfactuals(x, total_CFs=1, desired_class=desired,
                                                        proximity_weight=dice_proximity_weight,
                                                        diversity_weight=0.0,
                                                        features_to_vary=dice_features_to_vary,
                                                        posthoc_sparsity_param=dice_posthoc_sparsity_param)
                    else:
                        if self.num_classes == 2:
                            exp = e.generate_counterfactuals(x, total_CFs=1, desired_class="opposite",
                                                            proximity_weight=dice_proximity_weight,
                                                            diversity_weight=0.0,
                                                            features_to_vary=dice_features_to_vary,
                                                            posthoc_sparsity_param=dice_posthoc_sparsity_param)
                        else:
                            # multiclass but no explicit target: let DiCE choose
                            exp = e.generate_counterfactuals(x, total_CFs=1,
                                                            proximity_weight=dice_proximity_weight,
                                                            diversity_weight=0.0,
                                                            features_to_vary=dice_features_to_vary,
                                                            posthoc_sparsity_param=dice_posthoc_sparsity_param)
                    cf = json.loads(exp.to_json())
                except Exception:
                    cf = {'cfs_list': [None]}
                    print('No counterfactuals found')

                cfs_list = []
                for cfs in cf['cfs_list']:
                    if cfs is None:
                        continue
                    else:
                        cfs_list.append(np.array(cfs[0]).astype(np.float32))

                cfs_df = pd.DataFrame(cfs_list, columns=dataframe.columns)
                cfs_df = self.get_labeled_cfs(cfs=cfs_df.drop(columns=[out_name]), out_name=out_name)
                return cfs_df

            self.generate_counterfactuals = generate_counterfactuals

    def get_labeled_cfs(self, cfs:pd.DataFrame, out_name:str):
        if len(cfs) > 0:
            if self.cf_label == 'prediction':
                preds = self.model.predict(cfs)
                if preds.ndim == 1 or (hasattr(preds, 'shape') and preds.shape[1] == 1):
                    cf_preds = (preds >= 0.5).astype(np.float32)
                    # keep original binary scheme where CF label is centered at 0.5
                    cfs[out_name] = cf_preds - 0.5
                else:
                    labels = np.argmax(preds, axis=1)
                    cfs[out_name] = labels
            elif isinstance(self.cf_label, (int, float)):
                # Allow both positive labels (0.5 for binary) and negative out-of-band markers (-1 for multiclass)
                cfs[out_name] = self.cf_label
            return cfs
        else:
            return pd.DataFrame([], columns=[*cfs.columns, out_name])

    def query_api(self, x):
        raw_preds = self.model.predict(x)
        # Build a single-column predictions DataFrame: binary -> 0/1 float; multiclass -> label ints
        if raw_preds.ndim == 1 or (hasattr(raw_preds, 'shape') and raw_preds.shape[1] == 1):
            predictions = (raw_preds >= 0.5).astype(np.float32).reshape(-1, 1)
            predictions_df = pd.DataFrame(predictions, columns=[self.out_name])
            is_multiclass = False
        else:
            labels = np.argmax(raw_preds, axis=1)
            predictions_df = pd.DataFrame(labels, columns=[self.out_name])
            is_multiclass = True

        print(f'query API type: {self.method}')
        if self.method=='dualcfx':
            w = x
            counterfacts = self.generate_counterfactuals(w)
            countrcounts = self.generate_counterfactuals(counterfacts.drop(self.out_name, axis=1))
            results = pd.concat([x, predictions_df], axis=1)
            print(f'cf len:{len(counterfacts)}, ccf len:{len(countrcounts)}, res len:{len(results)}')
            results = pd.concat([results, counterfacts, countrcounts], axis=0)
            print(f'total len:{len(results)}')
        elif self.method=='dualcf':
            w = x
            counterfacts = self.generate_counterfactuals(w)
            countrcounts = self.generate_counterfactuals(counterfacts.drop(self.out_name, axis=1))
            results = pd.concat([counterfacts, countrcounts], axis=0)
        elif self.method=='onesided':
            if is_multiclass:
                # Interpret 'onesided' as generating CFs for samples predicted as class 0
                w = x.loc[predictions_df[self.out_name] == 0]
            else:
                w = x.loc[predictions_df[self.out_name] < 0.5]
            counterfacts = self.generate_counterfactuals(w)
            results = pd.concat([x, predictions_df], axis=1)
            results = pd.concat([results, counterfacts], axis=0)
        elif self.method=='twosidedcfonly':
            w = x
            results = self.generate_counterfactuals(w)
        else:
            w = x
            counterfacts = self.generate_counterfactuals(w)
            results = pd.concat([x, predictions_df], axis=1)
            results = pd.concat([results, counterfacts], axis=0)
        return results
    

def define_models(dataframe, targ_arch, surr_archs, num_classes: int = 2):
    targ_reg_coef = 0.001
    surr_reg_coef = 0.001

    # infer input dimension (exclude target column if present)
    input_dim = len(dataframe.columns) if dataframe is not None else None
    t_in = keras.layers.Input((input_dim,), dtype=tf.float32)
    t = t_in

    for layer_size in targ_arch:
        t = keras.layers.Dense(units=layer_size,
                               activation='relu',
                               kernel_regularizer=keras.regularizers.L2(l2=targ_reg_coef),
                               )(t)

    # target and surrogate model output shape depends on num_classes
    # By default keep binary output; callers can set num_classes>2 for multiclass
    def _build_output_layer(x, num_classes=2, reg_coef=0.0):
        if num_classes == 2:
            return keras.layers.Dense(units=1,
                                      activation='sigmoid',
                                      kernel_regularizer=keras.regularizers.L2(l2=reg_coef),
                                      )(x)
        else:
            return keras.layers.Dense(units=num_classes,
                                      activation='softmax',
                                      kernel_regularizer=keras.regularizers.L2(l2=reg_coef),
                                      )(x)

    # default num_classes for binary
    targ_output = _build_output_layer(t, num_classes=num_classes, reg_coef=targ_reg_coef)
    targ_model = keras.Model(inputs=t_in, outputs=t_output if False else targ_output)

    surr_models = []

    for surr_arch in surr_archs:
        s_in = keras.layers.Input((input_dim,), dtype=tf.float32)
        s = s_in
        for layer_size in surr_arch:
            s = keras.layers.Dense(units=layer_size,
                                   activation='relu',
                                   kernel_regularizer=keras.regularizers.L2(l2=surr_reg_coef),
                                   )(s)
    s_out = _build_output_layer(s, num_classes=num_classes, reg_coef=surr_reg_coef)
    surr_model = keras.Model(inputs=s_in, outputs=s_out)
    surr_models.append(surr_model)

    return targ_model, surr_models


# generate target models and data
def generate_query_data(exp_dir,
                        dataset,
                        use_balanced_df,
                        query_batch_size,
                        query_gen_method,
                        cf_method,
                        cf_generator,
                        cf_norm,
                        dice_backend,
                        dice_method,
                        num_queries,
                        ensemble_size,
                        targ_arch,
                        targ_epochs,
                        targ_lr,
                        surr_archs,
                        surr_epochs,
                        surr_lr,
                        imp_smart,
                        imp_naive,
                        batch_size,
                        dice_proximity_weight=1.5,
                        dice_posthoc_sparsity_param=0.1,
                        dice_features_to_vary='all',
                        knn_k=1,
                        roar_lambda=0.01, 
                        roar_delta_max=0.1,
                        cf_label=None,  # None = auto-select: 0.5 for binary, 'prediction' for multiclass
                        loss_type='onesidemod',
                        min_target_accuracy=0.6,
                        attack_set_balance=None,
                        cf_target_class=None,
                        num_classes: int = 2,
                        sample_limit: int = None
                        ):

    is_exist = os.path.exists(exp_dir)
    if not is_exist:
        os.makedirs(exp_dir)
        print(f'{exp_dir} created')
    else:
        print(f'{exp_dir} already exists')
        # exp_dir = f'{exp_dir}_{np.random.randint(100,999)}'
        now = datetime.now()
        now_str = now.strftime('%y%m%d%H%M')
        exp_dir = f'{exp_dir}_{now_str}'
        os.makedirs(exp_dir)
        print(f'created {exp_dir} instead')

    dataset_obj = ProcessedDataset(dataset, sample_limit=sample_limit)
    x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()

    # infer number of classes from dataset
    try:
        unique_labels = np.unique(pd.concat([y_trn, y_tst]))
        num_classes = int(len(unique_labels))
    except Exception:
        num_classes = 2

    # Auto-select cf_label if not specified: 0.5 for binary, 'prediction' for multiclass
    if cf_label is None:
        if num_classes == 2:
            cf_label = 0.5
            print('Using cf_label=0.5 (binary CF marker)')
        else:
            cf_label = 'prediction'
            print(f'Using cf_label="prediction" for multiclass (num_classes={num_classes})')
    elif cf_label == 'out-of-band':
        # Use out-of-band markers: 0.5 for binary, -1 for multiclass
        if num_classes == 2:
            cf_label = 0.5
            print('Using cf_label=0.5 (binary out-of-band CF marker)')
        else:
            cf_label = -1
            print(f'Using cf_label=-1 (multiclass out-of-band CF marker, num_classes={num_classes})')

    targ_model, surr_models = define_models(x_trn, targ_arch, surr_archs, num_classes=num_classes)
    surrmodellen = len(surr_models)

    print(f'number of surrogate models: {surrmodellen}')

    if len(imp_smart) != surrmodellen:
        print(f'imp_smart length is {len(imp_smart)}, but no of surrogate models is {surrmodellen}')
    
    np.save('{}/imp_naive'.format(exp_dir), np.array(imp_naive))
    np.save('{}/imp_smart'.format(exp_dir), np.array(imp_smart))

    # select appropriate loss/metrics for target model
    if num_classes == 2:
        targ_loss = keras.losses.BinaryCrossentropy(from_logits=False)
        targ_metrics = [keras.metrics.BinaryAccuracy(threshold=0.5)]
    else:
        targ_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        targ_metrics = [keras.metrics.SparseCategoricalAccuracy()]

    compile_models([targ_model],
                losses=[targ_loss],
                optimizers=[keras.optimizers.Adam(learning_rate=targ_lr)],
                metrics=targ_metrics)

    naive_models, smart_models = generate_test_models(surr_models)

    naive_losses = []
    smart_losses = []
    for m in range(surrmodellen):
        if num_classes == 2:
            naive_losses.append(get_modified_loss_fn(keras.losses.BinaryCrossentropy(from_logits=False), imp_naive[m], loss_type=loss_type, num_classes=num_classes))
            smart_losses.append(get_modified_loss_fn(keras.losses.BinaryCrossentropy(from_logits=False), imp_smart[m], loss_type=loss_type, num_classes=num_classes))
        else:
            # For multiclass with out-of-band CF markers (cf_label=-1), use CF-aware loss
            # Otherwise use standard SparseCategoricalCrossentropy
            base_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            naive_losses.append(get_modified_loss_fn(base_loss, imp_naive[m], loss_type=loss_type, num_classes=num_classes))
            smart_losses.append(get_modified_loss_fn(base_loss, imp_smart[m], loss_type=loss_type, num_classes=num_classes))

    if num_classes == 2:
        surr_metrics = [keras.metrics.BinaryAccuracy(threshold=0.5)]*surrmodellen
    else:
        surr_metrics = [keras.metrics.SparseCategoricalAccuracy()]*surrmodellen

    compile_models(naive_models,
                losses=naive_losses,
                optimizers=[keras.optimizers.Adam(learning_rate=surr_lr)]*surrmodellen,
                metrics=surr_metrics)
    compile_models(smart_models, 
                losses=smart_losses,
                optimizers=[keras.optimizers.Adam(learning_rate=surr_lr)]*surrmodellen,
                metrics=surr_metrics)

    for m in range(surrmodellen):
        naive_models[m].save('{}/naive_model_{:02d}.keras'.format(exp_dir, m))
        smart_models[m].save('{}/smart_model_{:02d}.keras'.format(exp_dir, m))

    info_list = [query_batch_size, query_gen_method, num_queries, ensemble_size, targ_epochs, surr_epochs, batch_size, \
                dataset, len(smart_models)]
    info_cols = ['query_batch_size', 'query_gen_method', 'num_queries', 'ensemble_size', 'targ_epochs', \
                'surr_epochs', 'batch_size', 'dataset', 'num_models']
    info_df = pd.DataFrame(np.array(info_list).reshape([1,len(info_list)]), 
                            columns=info_cols)
    info_df.to_csv('{}/info.csv'.format(exp_dir))

    print('generating query data')
    for i in range(ensemble_size):
        if use_balanced_df:
            x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()
        else:
            if attack_set_balance is None:
                raise Exception('sample ratio not specified while using unbalanced attack set')
            x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits(attack_balance=attack_set_balance)

        target_accuracy = 0.0
        attempt = 0
        while target_accuracy < min_target_accuracy and attempt < 100:
            print(f'target training attempt {attempt}')
            seed_layers = np.random.randint(100)
            tf.random.set_seed(np.random.randint(100))
            reset_weights([targ_model], seed=seed_layers)
            train_models([targ_model], x_trn=x_trn, y_trn=y_trn, epochs=targ_epochs, verbose=1)
            target_accuracy = evaluate_models([targ_model], x_tst, y_tst, num_classes=num_classes)[0][0]
            attempt += 1

        print(f'sample: {i} targ_accuracy:{target_accuracy}')
        targ_model.save('{}/targ_model_{:03d}.keras'.format(exp_dir, i))

        query_api = Query_API(model=targ_model, dataframe=dataframe, cts_features=numcols, 
                              out_name=targetcol, method=cf_method, generator=cf_generator, norm=cf_norm, 
                              dice_backend=dice_backend, dice_method=dice_method,
                              dice_proximity_weight=dice_proximity_weight,
                              dice_posthoc_sparsity_param=dice_posthoc_sparsity_param,
                              dice_features_to_vary=dice_features_to_vary,
                              knn_k=knn_k, roar_lambda=roar_lambda, roar_delta_max=roar_delta_max, 
                              cf_label=cf_label, cf_target_class=cf_target_class, num_classes=num_classes)

        query_gen = Query_Gen(x_atk, catcols, numcols)

        for j in range(num_queries):
            queries = query_gen.generate_queries(query_batch_size, method=query_gen_method)
            results = query_api.query_api(queries)
            results.to_csv('{}/query_{:03d}_{:03d}.csv'.format(exp_dir,i,j))

    return exp_dir


def generate_stats(exp_dir, pop_noncf=True, noise_sigma=0, loss_type='onesidemod'):
    print(f'generating stats from exp_dir: {exp_dir}')

    info_df = pd.read_csv('{}/info.csv'.format(exp_dir), index_col=0)

    dataset = str(info_df['dataset'][0])
    dataset_obj = ProcessedDataset(dataset)
    x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()
    
    # Infer number of classes from dataset
    try:
        unique_labels = np.unique(pd.concat([y_trn, y_tst]))
        num_classes = int(len(unique_labels))
    except Exception:
        num_classes = 2
    
    num_models = int(info_df['num_models'][0])
    naive_models = []
    smart_models = []
    imp_naive = np.load('{}/imp_naive.npy'.format(exp_dir))
    imp_smart = np.load('{}/imp_smart.npy'.format(exp_dir))
    print(f'imp_naive: {imp_naive}, imp_smart: {imp_smart}')

    for m in range(num_models):
        if num_classes == 2:
            base_loss = keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            base_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            
        naive_models.append(keras.models.load_model('{}/naive_model_{:02d}.keras'.format(exp_dir, m), \
            custom_objects={'loss_fn':get_modified_loss_fn(base_loss, imp_naive[m], loss_type=loss_type, num_classes=num_classes)}))
        smart_models.append(keras.models.load_model('{}/smart_model_{:02d}.keras'.format(exp_dir, m), \
            custom_objects={'loss_fn':get_modified_loss_fn(base_loss, imp_smart[m], loss_type=loss_type, num_classes=num_classes)}))

    query_batch_size = int(info_df['query_batch_size'][0])
    query_gen_method = str(info_df['query_gen_method'][0])
    num_queries = int(info_df['num_queries'][0])
    ensemble_size = int(info_df['ensemble_size'][0])
    targ_epochs = int(info_df['targ_epochs'][0])
    surr_epochs = int(info_df['surr_epochs'][0])
    surr_batch_size = int(info_df['batch_size'][0])

    query_gen = Query_Gen(x_atk, catcols, numcols)

    fid_naive = []
    fid_smart = []
    fid_uni_naive = []
    fid_uni_smart = []
    acc_naive = []
    acc_smart = []
    for i in range(ensemble_size):
        x_trn, y_trn, x_tst, y_tst, x_atk, y_atk, dataframe, numcols, catcols, targetcol = dataset_obj.get_splits()

        seed_layers = np.random.randint(100)
        tf.random.set_seed(np.random.randint(100))

        targ_model = keras.models.load_model('{}/targ_model_{:03d}.keras'.format(exp_dir, i))
        print(f'sample: {i} targ_accuracy: {evaluate_models([targ_model], x_tst, y_tst)[0][0]}')

        results = pd.read_csv('{}/query_{:03d}_{:03d}.csv'.format(exp_dir,i,0), index_col=0)

        if noise_sigma > 0:
            results = add_noise(results, targetcol, numcols, targ_model, pop_noncf=pop_noncf, sigma=noise_sigma)
        results_y = results[targetcol]
        results_x = results.drop(targetcol, axis=1)

        fid_naive_ensemble = []
        fid_smart_ensemble = []
        fid_uni_naive_ensemble = []
        fid_uni_smart_ensemble = []
        acc_naive_ensemble = []
        acc_smart_ensemble = []
        for j in range(num_queries):
            reset_weights(naive_models, seed=seed_layers)
            reset_weights(smart_models, seed=seed_layers)

            # print('dataset shape:', results_x.shape)
            naive_hist = train_models(naive_models, x_trn=results_x, y_trn=results_y, epochs=surr_epochs, batch_size=surr_batch_size)
            smart_hist = train_models(smart_models, x_trn=results_x, y_trn=results_y, epochs=surr_epochs, batch_size=surr_batch_size)

            # accq_n = evaluate_models(naive_models, results_x, results_y)[0]
            # accq_s = evaluate_models(smart_models, results_x, results_y)[0]
            # print('naive acc over queries:', accq_n)
            # print('smart acc over queries:', accq_s)
            accq_n, fid_n = evaluate_models(naive_models, x_tst, y_tst, targ_model=targ_model)
            fid_n = np.array(fid_n)
            accq_s, fid_s = evaluate_models(smart_models, x_tst, y_tst, targ_model=targ_model)
            fid_s = np.array(fid_s)
            # print('naive:', i, j, 'fidelity:', fid_n, 'accuracy:', acc_n)
            # print('smart:', i, j, 'fidelity:', fid_s, 'accuracy:', acc_s)
            # print('diff:', fid_s-fid_n)
            
            uni_queries = query_gen.generate_queries(10000)
            fid_uni_naive_ensemble.append(evaluate_models(naive_models, uni_queries, targ_model=targ_model)[1])
            fid_uni_smart_ensemble.append(evaluate_models(smart_models, uni_queries, targ_model=targ_model)[1])

            fid_naive_ensemble.append(fid_n)
            fid_smart_ensemble.append(fid_s)
            acc_naive_ensemble.append(accq_n)
            acc_smart_ensemble.append(accq_s)

            if j < num_queries-1:
                results = pd.read_csv('{}/query_{:03d}_{:03d}.csv'.format(exp_dir,i,j+1), index_col=0)
                
                if noise_sigma > 0:
                    results = add_noise(results, targetcol, numcols, targ_model, pop_noncf=pop_noncf, sigma=noise_sigma)
                
                new_y = results[targetcol]
                new_x = results.drop(targetcol, axis=1)

                results_y = pd.concat([new_y, results_y])
                results_x = pd.concat([new_x, results_x])
                
        fid_naive.append(fid_naive_ensemble)
        fid_smart.append(fid_smart_ensemble)
        fid_uni_naive.append(fid_uni_naive_ensemble)
        fid_uni_smart.append(fid_uni_smart_ensemble)
        acc_naive.append(acc_naive_ensemble)
        acc_smart.append(acc_smart_ensemble)

    fid_naive = np.array(fid_naive)
    fid_smart = np.array(fid_smart)
    fid_uni_naive = np.array(fid_uni_naive)
    fid_uni_smart = np.array(fid_uni_smart)
    acc_naive = np.array(acc_naive)
    acc_smart = np.array(acc_smart)

    # Use np.save to preserve array dimensionality (supports >2D) and match .npy expectations
    np.save(f'{exp_dir}/fid_naive.npy', fid_naive)
    np.save(f'{exp_dir}/fid_smart.npy', fid_smart)
    np.save(f'{exp_dir}/fid_uni_naive.npy', fid_uni_naive)
    np.save(f'{exp_dir}/fid_uni_smart.npy', fid_uni_smart)
    np.save(f'{exp_dir}/acc_naive.npy', acc_naive)
    np.save(f'{exp_dir}/acc_smart.npy', acc_smart)


def add_noise(query_df, targetcol, numcols, targ_model, pop_noncf=True, sigma=0):
    cf_df = query_df[query_df[targetcol]==0.5]
    query_df = query_df[query_df[targetcol]!=0.5]

    if len(cf_df) > 0:
        cf_df[numcols] = cf_df[numcols] + np.random.normal(0, sigma, cf_df[numcols].shape)
        
        preds = targ_model.predict(cf_df[cf_df.columns[:-1]])
        preds = (preds >= 0.5)
    
        if pop_noncf:
            cf_df = cf_df[preds]
        query_df = pd.concat([query_df, cf_df], ignore_index=True)
        
    return query_df


import time
class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def end_and_write_to_file(self, filepath, display=True):
        time_elapsed = time.time()-self.start_time

        if display:
            print(f'----------------------------------')
            print(f'----------------------------------')
            print(f'{time_elapsed} seconds elapsed')
            print(f'----------------------------------')

        with open(f'{filepath}/execution_time.txt', 'w') as file:
            file.write(f'Execution Time: {time_elapsed} seconds')
