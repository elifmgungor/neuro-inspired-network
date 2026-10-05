'''
Compare models across seeds.
Reads the per-run tables and writes mean, standard deviation and number of seeds
for every model, and paired differences between models trained with the same seed.

Outputs:
    results/tables/model_comparison.csv           test accuracy per model and training-set size
    results/tables/paired_differences.csv         difference between two models, per training-set size and seed
    results/tables/paired_comparison.csv          the paired differences summarized over seeds
    results/tables/noise_comparison.csv           test accuracy per model and noise level
    results/tables/representation_comparison.csv  hidden-layer measures per model
'''
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIRECTORY = PROJECT_ROOT/'results/tables'

# Pairs of models to compare (model, reference) and what the difference measures
COMPARISONS = [
    ('dendritic', 'baseline', 'dendritic branches vs standard neuron'),
    ('linear_branches', 'baseline', 'splitting inputs into linear branches'),
    ('dale', 'baseline', "Dale's law"),
    ('divisive_norm', 'baseline', 'divisive normalization'),
    ('dendritic', 'linear_branches', 'branch nonlinearity (branches see all inputs)'),
    ('dendritic_sparse', 'linear_branches_sparse', 'branch nonlinearity (16 inputs per branch)'),
    ('dendritic_sparse', 'dendritic', 'restricted sampling (nonlinear branches)'),
    ('linear_branches_sparse', 'linear_branches', 'restricted sampling (linear branches)'),
    ('dendritic_sparse', 'baseline', 'sparse dendritic neuron vs standard neuron'),
    ('mlp_dendritic_sparse', 'mlp_vanilla', 'MLP: dendritic vs vanilla (Chavlis & Poirazi, 2025)'),
    ('mlp_dendritic_sparse', 'mlp_linear_branches_sparse', 'MLP: dendritic nonlinearity (16 inputs per dendrite)'),
    ('mlp_linear_branches_sparse', 'mlp_vanilla', 'MLP: linear dendrites vs vanilla'),
]

def mean_std_n(table, group_columns, value_columns):
    '''
    Mean, standard deviation and number of seeds of value_columns for every group.
    '''
    result = table.groupby(group_columns)[value_columns].agg(['mean', 'std', 'count']).round(4)
    result.columns = [f'{value}_{statistic}' for value, statistic in result.columns]
    return result.reset_index()

def paired_differences(summary):
    '''
    Test accuracy of model minus test accuracy of reference, for every
    training-set size and seed where both models were trained.
    '''
    rows = []
    for model, reference, meaning in COMPARISONS:
        a = summary[summary['model'] == model].set_index(['train_percent', 'seed'])['test_accuracy']
        b = summary[summary['model'] == reference].set_index(['train_percent', 'seed'])['test_accuracy']
        for (train_percent, seed), difference in (a - b).dropna().items():
            rows.append({'model': model, 'reference': reference, 'meaning': meaning,
                         'train_percent': train_percent, 'seed': seed, 'difference': round(difference, 4)})
    return pd.DataFrame(rows)

def main():
    # 1. Test accuracy for every model and training-set size
    summary = pd.read_csv(TABLES_DIRECTORY/'summary.csv')
    comparison = mean_std_n(summary, ['model', 'train_percent'], ['test_accuracy', 'test_balanced_accuracy'])
    comparison.to_csv(TABLES_DIRECTORY/'model_comparison.csv', index=False)
    print(comparison.to_string(index=False), '\n')

    # 2. Paired differences: the same seed means the same data split, data order and convolution weights
    differences = paired_differences(summary)
    differences.to_csv(TABLES_DIRECTORY/'paired_differences.csv', index=False)
    paired = differences.groupby(['model', 'reference', 'meaning', 'train_percent'])['difference'].agg(
        mean='mean', std='std', n_seeds='count', n_positive=lambda d: int((d > 0).sum())).round(4).reset_index()
    paired.to_csv(TABLES_DIRECTORY/'paired_comparison.csv', index=False)
    print(paired.to_string(index=False), '\n')

    # 3. Robustness to noise
    noise_file = TABLES_DIRECTORY/'noise_robustness.csv'
    if noise_file.exists():
        noise = mean_std_n(pd.read_csv(noise_file), ['model', 'noise_std'], ['test_accuracy'])
        noise.to_csv(TABLES_DIRECTORY/'noise_comparison.csv', index=False)
        print(noise.to_string(index=False), '\n')

    # 4. Hidden-layer representations
    representations_file = TABLES_DIRECTORY/'representations.csv'
    if representations_file.exists():
        columns = ['active_fraction', 'dead_fraction', 'class_selectivity', 'rdm_similarity_to_baseline']
        representations = mean_std_n(pd.read_csv(representations_file), ['model'], columns)
        representations.to_csv(TABLES_DIRECTORY/'representation_comparison.csv', index=False)
        print(representations.to_string(index=False))

if __name__ == "__main__":
    main()
