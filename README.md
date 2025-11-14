# Model reconstruction using counterfactual explanations
This repository provides the code for the paper [*"Model reconstruction using counterfactual explanations: A perspective from polytope theory"*](https://arxiv.org/abs/2405.05369) by [Pasan Dissanayake](https://pasandissanayake.github.io/) and [Sanghamitra Dutta](https://sites.google.com/site/sanghamitraweb/) accepted at NeurIPS 2024.

## Experiments
### Setup
```bash
pip install -r requirements.txt
pip install foolbox adversarial-robustness-toolbox
```

### Running the experiments
The script `example.sh` contains convenience commands for running experiments; for programmatic runs see `main.py`.
```bash
python main.py --dir ./results/test --dataset heloc --use_balanced_df True --query_size 50 --cfgenerator mccf \
               --num_queries 8 --ensemble_size 50 --target_archi 20 10 --surr_archi 20 10
```

## Key Configuration Options

### Counterfactual Label Selection (`--cflabel`)
The system automatically selects appropriate counterfactual labels based on classification task:

- **Binary Classification**: Default `cf_label = 0.5` (decision boundary)
- **Multiclass Classification**: Default `cf_label = 'prediction'` (uses model prediction)
- **Out-of-Band Mode**: Use `--cflabel out-of-band` for multiclass CF-aware training
  - Binary: `cf_label = 0.5` (same as default)
  - Multiclass: `cf_label = -1` (special marker for masking)

**Options**:
- `auto` (default): Automatically selects 0.5 for binary, 'prediction' for multiclass
- `out-of-band`: Enables CF-aware loss for multiclass using -1 marker
- `prediction`: Uses target model predictions as CF labels
- Numeric value (e.g., `0.7`): Custom threshold for binary classification

### Loss Functions (`--loss_type`)
The implementation supports different loss strategies for binary and multiclass settings:

**Binary Classification**:
- `ordinary`: Standard binary cross-entropy
- `onesidemod`: CF-aware loss with penalty term for samples at threshold (default)
- `bcecf`: Binary cross-entropy with soft labels  
- `twosidemod`: CF-aware loss accounting for both sides of decision boundary

**Multiclass Classification (out-of-band mode)**:
- `ordinary`: Standard sparse categorical cross-entropy
- `onesidemod`: CF-aware loss that masks out CF samples (y=-1) and trains only on regular samples

The CF-aware loss differs between binary and multiclass:
- **Binary**: Applies penalty term encouraging predictions near threshold for CF samples
- **Multiclass**: Masks CF samples (y=-1) and excludes them from gradient computation

### Visualizing results
The experiments generate files containing the queries, models and statistics. To visualize the results, use the Jupyter Notebook `visualize.ipynb`. The directory [results](results) provides some results that are included in the paper.

## Acknowledgement
Our code uses the codebase from the paper *"Black, E., Wang, Z., Fredrikson, M., & Datta, A., Consistent Counterfactuals for Deep Models, ICLR 2021"* from [https://github.com/zifanw/consistency](https://github.com/zifanw/consistency).

## License
Please see [LICENSE](LICENSE).

## Multiclass Support & Testing

The codebase supports multiclass datasets (Iris, MNIST, CIFAR) with automatic CF label selection and CF-aware loss functions.

### Key Features
- **Automatic cf_label selection**: Binary tasks use 0.5 threshold, multiclass uses model predictions or out-of-band mode
- **CF-aware loss for multiclass**: Out-of-band mode uses -1 marker to mask CF samples during training
- **Multiclass datasets**: Built-in support for Iris, MNIST, CIFAR with optional subsampling
- **Test utilities**: Comprehensive test suite in `tests/` directory

### Additional Arguments
- `--cf_target_class` (int): Target class for CF generation (when supported by backend)
- `--num_classes` (int): Number of classes for multiclass datasets  
- `--sample_limit` (int): Subsample large datasets for quick testing

### Examples

**Binary classification with CF-aware loss:**
```bash
python main.py --dataset heloc --cflabel auto --loss_type onesidemod
```

**Multiclass with out-of-band CF-aware training:**
```bash
python main.py --dataset iris --cflabel out-of-band --loss_type onesidemod --num_classes 3
```

**Quick MNIST experiment with subsampling:**
```bash
python main.py --dataset mnist --num_classes 10 --sample_limit 1000 --ensemble_size 2
```

### Testing
Run the test suite to verify functionality:
```bash
# Activate virtual environment
source .venv/bin/activate

# Test CF label auto-selection logic
PYTHONPATH=$(pwd) python tests/test_cf_label_simple.py

# Test multiclass CF-aware loss (requires datasets)
PYTHONPATH=$(pwd) python tests/test_out_of_band_mode.py
```

### Utilities
- `tools/inspect_stats.py`: Examine .npy/.csv statistics files with human-readable output
- `tests/`: Comprehensive test suite for new functionality
