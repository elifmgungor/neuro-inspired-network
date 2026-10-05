'''
Robustness to input noise.
Every trained model (full training set) is tested on the test images
with increasing amounts of Gaussian noise added.

Output: results/tables/noise_robustness.csv
'''
import sys
from pathlib import Path

import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root/'src'))
from neuroinspired.data import load_cifar10
from neuroinspired.models import build_model
from neuroinspired.training import evaluate
from neuroinspired.results import get_run_name, update_table

# Noise standard deviation, in units of the per-channel pixel standard deviation
# (the images are already normalized, so 1.0 = noise as strong as the image contrast)
NOISE_LEVELS = [0, 0.25, 0.5, 1.0, 1.5]

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tables_directory = project_root/'results/tables'
    _, _, test_loader = load_cifar10(project_root/'data')

    summary = pd.read_csv(tables_directory/'summary.csv')
    runs = summary[summary['train_percent'] == 100]

    rows = []
    for _, run in runs.iterrows():
        run_name = get_run_name(run['model'], 100, run['seed'])
        model = build_model(run['model']).to(device)
        model.load_state_dict(torch.load(project_root/f'results/models/{run_name}.pt', map_location=device))

        for noise_std in NOISE_LEVELS:
            torch.manual_seed(0)  # the same noise for every model
            test = evaluate(model, test_loader, device, noise_std)
            rows.append({'model': run['model'], 'seed': run['seed'], 'noise_std': noise_std,
                         'test_accuracy': round(test['accuracy'], 4)})
            print(f"{run_name} | noise {noise_std:.2f} | test accuracy {test['accuracy']:.3f}")

    update_table(tables_directory/'noise_robustness.csv', rows, ['model', 'seed', 'noise_std'])
    print('Results saved to', tables_directory/'noise_robustness.csv')

if __name__ == "__main__":
    main()
