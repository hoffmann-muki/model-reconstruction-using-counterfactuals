import argparse
from utils_v8 import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Execute model reconstruction attack using counterfactual examples')
    parser.add_argument('--dir', type=str, default='./', help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='heloc', 
                        choices=['adultincome', 'dccc', 'compas', 'heloc', 'iris', 'mnist', 'cifar'], help='Dataset to use')
    parser.add_argument('--use_balanced_df', type=bool, default=True, help='Use a balanced attack set if True')
    parser.add_argument('--query_size', type=int, default=50, help='No. of datapoints in a single query')
    parser.add_argument('--cfmethod', type=str, default='onesided', 
                        choices=['onesided', 'twosidedcfonly', 'dualcf', 'dualcfx'], help='Regions which CFs are generated')
    parser.add_argument('--cfgenerator', type=str, default='mccf', 
                        choices=['mccf', 'knn', 'roar', 'dice', ''], help='Counterfactual generating method to use')
    parser.add_argument('--cfnorm', type=int, default=2, choices=[1, 2], help='CF cost function norm')
    parser.add_argument('--num_queries', type=int, default=8, help='Number of queries')
    parser.add_argument('--ensemble_size', type=int, default=50, 
                        help='Ensemble size to repeat the experiment and compute averages over')
    parser.add_argument('--target_archi', type=int, nargs='+', default=[20, 10], 
                        help='Target model architecture as a list of the sizes of intermediate layers')
    parser.add_argument('--target_epochs', type=int, default=200, help='Target model training epochs')
    parser.add_argument('--surr_archi', type=int, nargs='+', default=[[20, 10], [20, 10, 5]], 
                        help='Surrogate model architecture')
    parser.add_argument('--surr_epochs', type=int, default=200, help='Surrogate model training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--cflabel', type=str, default='auto', 
                        help=('Label to use for counterfactual explanations in the query results; '
                              'can be a float (e.g., 0.5 for binary), \'prediction\' to use target model output, '
                              '\'out-of-band\' to use -1 marker with CF-aware loss for multiclass, '
                              'or \'auto\' (default: 0.5 for binary, prediction for multiclass)'))
    parser.add_argument('--loss_type', type=str, default='onesidemod', 
                        choices=['onesidemod', 'ordinary', 'bcecf', 'twosidemod'],
                        help=('onesidemod: CCA loss as described in the paper; '
                              'ordinary: ordinary binary cross entropy loss with hard labels; '
                              'bcecf: binary cross entropy loss with soft labels; '
                              'twosidemod: CCA loss that accounts for CFs from both sides of the decision boundary'))
    # multiclass options
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (choose n>2 for multiclass datasets)')
    parser.add_argument('--sample_limit', type=int, default=None, help='Optional subsample limit for large datasets (MNIST/CIFAR)')
    parser.add_argument('--cf_target_class', type=int, default=None, help='Optional target class for counterfactual generation (int)')

    args = parser.parse_args()
    imp_naive = [-1]
    imp_smart = [0.5]
    
    # Parse cf_label: 'auto' -> None (auto-select), numeric -> float, else string
    if args.cflabel == 'auto':
        cf_label = None
    elif args.cflabel in ['out-of-band', 'prediction']:
        cf_label = args.cflabel
    else:
        try:
            cf_label = float(args.cflabel)
        except ValueError:
            cf_label = args.cflabel

    timer = Timer()
    timer.start()
    exp_dir = generate_query_data(exp_dir=args.dir,
                        dataset=args.dataset,
                        use_balanced_df=args.use_balanced_df,
                        query_batch_size=args.query_size,
                        query_gen_method='naivedat',
                        cf_method=args.cfmethod,
                        dice_backend='TF2',
                        dice_method='random',
                        cf_generator=args.cfgenerator,
                        cf_norm=args.cfnorm,
                        num_queries=args.num_queries,
                        ensemble_size=args.ensemble_size,
                        targ_arch=args.target_archi,
                        targ_epochs=args.target_epochs,
                        targ_lr=0.01,
                        surr_archs=[args.surr_archi],
                        surr_epochs=args.surr_epochs,
                        surr_lr=0.01,
                        imp_smart=imp_smart,
                        imp_naive=imp_naive,
                        batch_size=args.batch_size,
                        cf_label=cf_label,
                        loss_type=args.loss_type,
                        cf_target_class=args.cf_target_class,
                        num_classes=args.num_classes,
                        sample_limit=args.sample_limit
                    )
    generate_stats(exp_dir, loss_type=args.loss_type)
    timer.end_and_write_to_file(exp_dir)