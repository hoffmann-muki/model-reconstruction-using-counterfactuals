#!/bin/bash

python main.py --dir ./results/test \
               --dataset heloc \
               --use_balanced_df True \
               --query_size 50 \
               --cfgenerator knn \
               --num_queries 8 \
               --ensemble_size 2 \
               --target_archi 20 10 \
               --surr_archi 20 10