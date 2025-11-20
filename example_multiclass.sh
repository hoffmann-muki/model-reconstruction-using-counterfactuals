#!/bin/bash

# Generate multiclass counterfactual queries for Iris dataset (3 classes)
# This script corresponds to the multiclass surrogate model test

python main.py --dir ./results/test_multiclass_surrogate \
               --dataset iris \
               --use_balanced_df True \
               --query_size 8 \
               --cfmethod onesided \
               --cfgenerator knn \
               --cfnorm 2 \
               --num_queries 1 \
               --ensemble_size 1 \
               --target_archi 20 10 \
               --target_epochs 8 \
               --surr_archi 10 \
               --surr_epochs 12 \
               --batch_size 32 \
               --cflabel auto \
               --loss_type onesidemod \
               --num_classes 3 \
               --cf_target_class 2
