# Model reconstruction using counterfactual explanations
This repository provides the code for the paper [*"Model reconstruction using counterfactual explanations: A perspective from polytope theory"*](https://arxiv.org/abs/2405.05369) by [Pasan Dissanayake](https://pasandissanayake.github.io/) and [Sanghamitra Dutta](https://sites.google.com/site/sanghamitraweb/) accepted at NeurIPS 2024.

## Experiments
### Setup
```bash
pip install foolbox
pip install adversarial-robustness-toolbox
pip install -r requirements.txt
```

### Running the experiments
The script `examples.sh` contains a Bash script for running experiments. For more options, look into `main.py`.
```bash
python main.py --dir ./results/test --dataset heloc --use_balanced_df True --query_size 50 --cfgenerator mccf \
               --num_queries 8--ensemble_size 50 --target_archi 20 10 --surr_archi 20 10
```

### Visualizing results
The experiments generate files containing the queries, models and statistics. To visualize the results, use the Jupyter Notebook `visualize.ipynb`. The directory [results](results) provides some results that are included in the paper.

## Acknowledgement
Our code uses the codebase from the paper *"Black, E., Wang, Z., Fredrikson, M., & Datta, A., Consistent Counterfactuals for Deep Models, ICLR 2021"* from [https://github.com/zifanw/consistency](https://github.com/zifanw/consistency).

## License
Please see [LICENSE](LICENSE).

## Multiclass & Testing (new)

The codebase now supports multiclass datasets (Iris, MNIST, CIFAR) and includes helpers for testing and small smoke runs.

Key flags and utility functions:
- `cf_target_class` (int or None): when provided, counterfactual backends that support class-targeted CFs (e.g., DiCE) will request CFs for that target class. If not provided, binary tasks use `desired_class='opposite'` and multiclass tasks let the backend decide.
- `num_classes` (int): pass the number of classes (useful when calling `generate_query_data` directly).
- `sample_limit` (int or None): when loading large datasets (MNIST/CIFAR) you can pass a `sample_limit` to subsample the dataset for quick smoke runs.

Examples (run in project root, activate the virtualenv first):

Run a quick Iris smoke experiment (multiclass):
```bash
python - <<'PY'
from utils_v8 import generate_query_data
generate_query_data('results/test_smoke_iris', 'iris', True, 2, 'naivedat', 'onesided', 'knn', 2, 'TF', 'random', 1, 1,
                    [16], 3, 0.01, [[8]], 2, 0.01, [-1], [-1], 16, cf_target_class=1)
PY
```

Run a subsampled MNIST smoke experiment (500 samples):
```bash
python - <<'PY'
from utils_v8 import generate_query_data
generate_query_data('results/test_smoke_mnist', 'mnist', True, 2, 'naivedat', 'onesided', 'knn', 2, 'TF', 'random', 1, 1,
                    [32], 3, 0.01, [[16]], 2, 0.01, [-1], [-1], 16, cf_target_class=1, num_classes=10, sample_limit=500)
PY
```

Caveats and notes:
- KNN now builds per-class pools and a global fallback search to return *real samples* as fallback counterfactuals (preferred over centroids). This increases the chance a fallback CF is actionable and labeled appropriately.
- DiCE receives `desired_class=int(cf_target_class)` when provided. For binary tasks, `desired_class='opposite'` is used by default.
- ROAR/IterativeSearch are passed `num_classes` where applicable; ROAR uses a nearest-neighbor fallback that returns a real sample if the recourse solver fails for multiclass targets.
- The code emits pandas FutureWarnings around concat operations; they are non-fatal.

Tests:
- A small pytest `tests/test_iris_smoke.py` is included as a smoke test. Run it with:
```bash
PYTHONPATH=. pytest -q tests/test_iris_smoke.py
```

