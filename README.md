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
