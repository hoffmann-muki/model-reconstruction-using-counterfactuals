from utils_v8 import generate_stats, generate_query_data

print("=== Binary Surrogate Model Test ===")

# Generate some query data to create surrogate models from the target model
print("Generating counterfactual queries...")
exp_dir = generate_query_data(exp_dir='results/test_binary_surrogate',
                                dataset='heloc',
                                use_balanced_df=True,
                                query_batch_size=50,
                                query_gen_method='naivedat',
                                cf_method='onesided',
                                cf_generator='knn',
                                cf_norm=2,
                                dice_backend='TF2',
                                dice_method='random',
                                num_queries=8,
                                ensemble_size=1,
                                targ_arch=[20, 10],
                                targ_epochs=10,
                                targ_lr=0.01,
                                surr_archs=[[20, 10], [20, 10, 5]],
                                surr_epochs=10,
                                surr_lr=0.01,
                                cf_label=0.5,
                                imp_naive=[-1],
                                imp_smart=[0.5],
                                batch_size=32,
                                cf_target_class=1,
                                num_classes=2
                            )

print(f"Query data generated in: {exp_dir}")

# Get summary statistics from the experiment
generate_stats(exp_dir, loss_type='onesidemod')