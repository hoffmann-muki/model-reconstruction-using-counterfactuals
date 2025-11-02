import argparse
from utils_v8 import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Execute model reconstruction attack using counterfactual examples')
    parser.add_argument('--dir', type=str, default='./', help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='heloc', 
                        choices=['adultincome', 'dccc', 'compas', 'heloc'], help='Dataset to use')
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
    parser.add_argument('--cflabel', type=str, default='0.5', 
                        help=('Label to use for counterfactual explanations in the query results; '
                              'can be a float in [0,1] or \'prediction\' to use the softmax output of the target'))
    parser.add_argument('--loss_type', type=str, default='onesidemod', 
                        choices=['onesidemod', 'ordinary', 'bcecf', 'twosidemod'],
                        help=('onesidemod: CCA loss as described in the paper; '
                              'ordinary: ordinary binary cross entropy loss with hard labels; '
                              'bcecf: binary cross entropy loss with soft labels; '
                              'twosidemod: CCA loss that accounts for CFs from both sides of the decision boundary'))

    args = parser.parse_args()
    imp_naive = [-1]
    imp_smart = [0.5]
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
                        loss_type=args.loss_type
                    )
    generate_stats(exp_dir, loss_type=args.loss_type)
    timer.end_and_write_to_file(exp_dir)