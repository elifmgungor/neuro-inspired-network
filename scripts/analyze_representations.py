'''
What does the hidden layer learn?
For every trained model (full training set), the hidden-layer activity on the
test images is summarized with simple, interpretable measures:
    - active_fraction:     average fraction of hidden neurons active (> 0) per image
    - dead_fraction:       fraction of hidden neurons never active on the test set
    - class_selectivity:   how strongly neurons prefer one class (0 = no preference, 1 = only one class)
    - rdm_similarity_to_baseline: how similar the class-by-class representational geometry
                           is to the baseline trained with the same seed (1 = identical)

Outputs:
    results/tables/representations.csv
    results/tables/<run>_class_means.csv   (mean activity of each neuron for each class)
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root/'src'))
from neuroinspired.data import CLASSES, load_cifar10
from neuroinspired.models import MLP_MODELS, build_model
from neuroinspired.training import get_hidden_activations
from neuroinspired.results import get_run_name, update_table

def class_selectivity(class_means):
    '''
    Class selectivity index per neuron (Morcos et al., 2018):
        (mean for the preferred class - mean for the other classes) / (sum of both)
    Neurons that are never active are ignored.
    '''
    preferred = class_means.max(axis=0)
    others = (class_means.sum(axis=0) - preferred) / (len(class_means) - 1)
    active = preferred > 0
    return np.mean((preferred[active] - others[active]) / (preferred[active] + others[active]))

def representational_dissimilarity(class_means):
    '''
    Representational dissimilarity matrix (RDM): 1 - correlation between the
    average hidden-layer patterns of every pair of classes (classes x classes).
    '''
    return 1 - np.corrcoef(class_means)

def rdm_similarity(rdm_a, rdm_b):
    '''
    Correlation between the upper triangles of two RDMs.
    '''
    upper = np.triu_indices(len(rdm_a), k=1)
    return np.corrcoef(rdm_a[upper], rdm_b[upper])[0, 1]

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tables_directory = project_root/'results/tables'
    _, _, test_loader = load_cifar10(project_root/'data')

    summary = pd.read_csv(tables_directory/'summary.csv')
    runs = summary[summary['train_percent'] == 100]

    # 1. Hidden-layer activity of every model
    class_means = {}
    rows = []
    for _, run in runs.iterrows():
        run_name = get_run_name(run['model'], 100, run['seed'])
        model = build_model(run['model']).to(device)
        model.load_state_dict(torch.load(project_root/f'results/models/{run_name}.pt', map_location=device))
        activations, labels = get_hidden_activations(model, test_loader, device)

        means = np.array([activations[labels == c].mean(axis=0) for c in range(len(CLASSES))])
        class_means[(run['model'], run['seed'])] = means
        pd.DataFrame(means, index=CLASSES).to_csv(tables_directory/f'{run_name}_class_means.csv')

        rows.append({'model': run['model'], 'seed': run['seed'],
                     'active_fraction': round(float((activations > 0).mean()), 4),
                     'dead_fraction': round(float((activations.max(axis=0) == 0).mean()), 4),
                     'class_selectivity': round(float(class_selectivity(means)), 4)})

    # 2. Compare the representational geometry of every model with the baseline of the same seed
    #    (CNN models with 'baseline', MLP models with 'mlp_vanilla')
    for row in rows:
        reference = 'mlp_vanilla' if row['model'] in MLP_MODELS else 'baseline'
        baseline = class_means.get((reference, row['seed']))
        if baseline is None:
            row['rdm_similarity_to_baseline'] = np.nan
        else:
            own = class_means[(row['model'], row['seed'])]
            row['rdm_similarity_to_baseline'] = round(float(rdm_similarity(representational_dissimilarity(own), representational_dissimilarity(baseline))), 4)
        print(row)

    update_table(tables_directory/'representations.csv', rows, ['model', 'seed'])
    print('Results saved to', tables_directory/'representations.csv')

if __name__ == "__main__":
    main()
