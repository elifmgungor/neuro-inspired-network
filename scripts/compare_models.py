'''
Compare models across seeds.
Reads the per-run tables and writes mean, standard deviation and number of seeds
for every model.

Outputs:
    results/tables/model_comparison.csv           test accuracy per model and training-set size
    results/tables/noise_comparison.csv           test accuracy per model and noise level
    results/tables/representation_comparison.csv  hidden-layer measures per model
'''
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIRECTORY = PROJECT_ROOT/'results/tables'

def mean_std_n(table, group_columns, value_columns):
    '''
    Mean, standard deviation and number of seeds of value_columns for every group.
    '''
    result = table.groupby(group_columns)[value_columns].agg(['mean', 'std', 'count']).round(4)
    result.columns = [f'{value}_{statistic}' for value, statistic in result.columns]
    return result.reset_index()

def main():
    # 1. Test accuracy for every model and training-set size
    summary = pd.read_csv(TABLES_DIRECTORY/'summary.csv')
    comparison = mean_std_n(summary, ['model', 'train_percent'], ['test_accuracy', 'test_balanced_accuracy'])
    comparison.to_csv(TABLES_DIRECTORY/'model_comparison.csv', index=False)
    print(comparison.to_string(index=False), '\n')

    # 2. Robustness to noise
    noise_file = TABLES_DIRECTORY/'noise_robustness.csv'
    if noise_file.exists():
        noise = mean_std_n(pd.read_csv(noise_file), ['model', 'noise_std'], ['test_accuracy'])
        noise.to_csv(TABLES_DIRECTORY/'noise_comparison.csv', index=False)
        print(noise.to_string(index=False), '\n')

    # 3. Hidden-layer representations
    representations_file = TABLES_DIRECTORY/'representations.csv'
    if representations_file.exists():
        columns = ['active_fraction', 'dead_fraction', 'class_selectivity', 'rdm_similarity_to_baseline']
        representations = mean_std_n(pd.read_csv(representations_file), ['model'], columns)
        representations.to_csv(TABLES_DIRECTORY/'representation_comparison.csv', index=False)
        print(representations.to_string(index=False))

if __name__ == "__main__":
    main()
