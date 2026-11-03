'''
Train and evaluate one model on CIFAR-10.
Results are written to results/tables and the trained model to results/models.

Usage:
    python3 scripts/train.py <model> <seed> [train_percent]

Examples:
    python3 scripts/train.py baseline 1
    python3 scripts/train.py dendritic 1 10    (only 10% of the training images)
'''
import argparse
import copy
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root/'src'))
from neuroinspired.checkpoints import checkpoint_matches, run_metadata, select_device
from neuroinspired.data import load_cifar10
from neuroinspired.models import build_model
from neuroinspired.training import train_one_epoch, evaluate
from neuroinspired.results import get_run_name, save_run_results

# Early stopping (the same rule for every model): training stops when the validation
# accuracy has not improved for PATIENCE epochs, or after MAX_EPOCHS epochs.
# Both are given for the full training set and scaled up with fewer training images.
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('seed', type=int)
    parser.add_argument('train_percent', type=int, nargs='?', default=100, choices=range(1, 101))
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip only when checkpoint provenance matches this run')
    args = parser.parse_args()
    model_name, seed, train_percent = args.model, args.seed, args.train_percent
    run_name = get_run_name(model_name, train_percent, seed)

    # With less data an epoch has fewer weight updates, so allow proportionally more epochs
    # (e.g. 10% of the images: at most 1000 epochs, stop after 100 epochs without improvement)
    max_epochs = MAX_EPOCHS * 100 // train_percent
    patience = PATIENCE * 100 // train_percent

    torch.manual_seed(seed)
    device = select_device()
    config = {'model': model_name, 'seed': seed, 'train_percent': train_percent,
              'max_epochs': max_epochs, 'patience': patience, 'learning_rate': LEARNING_RATE,
              'optimizer': 'Adam', 'batch_size': 128, 'n_validation': 5000}
    metadata = run_metadata(project_root, config, device)
    model_file = project_root / 'results/models' / f'{run_name}.pt'
    if args.skip_existing and checkpoint_matches(model_file, metadata):
        print(f'Skipping {run_name}: matching checkpoint provenance')
        return
    if args.skip_existing and model_file.exists():
        print(f'Retraining {run_name}: checkpoint provenance missing or changed')

    # 1. Data
    train_loader, validation_loader, test_loader = load_cifar10(
        project_root/'data', train_percent, random_state=seed,
        batch_size=config['batch_size'], n_validation=config['n_validation'])

    # 2. Model and optimizer
    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'Run: {run_name}, device: {device}, parameters: {n_parameters:,}, training images: {len(train_loader.dataset):,}')

    # Reset the seed so every model sees the training images in the same order
    torch.manual_seed(seed)

    # 3. Training
    # The model of the epoch with the highest validation accuracy is kept,
    # so later epochs where the network starts to memorize the training images are not used.
    history = []
    best_accuracy, best_epoch = -1, 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        history.append({'epoch': epoch, 'train_loss': round(train_loss, 4), 'validation_accuracy': round(validation['accuracy'], 4)})
        print(f"Epoch {epoch:4d} | train loss {train_loss:.3f} | validation accuracy {validation['accuracy']:.3f}")

        if validation['accuracy'] > best_accuracy:
            best_accuracy, best_epoch = validation['accuracy'], epoch
            best_weights = copy.deepcopy(model.state_dict())

        if epoch - best_epoch >= patience:
            print(f'Early stopping: no improvement for {patience} epochs')
            break

    # 4. Final evaluation of the best model on the test set
    model.load_state_dict(best_weights)
    test = evaluate(model, test_loader, device)
    print()
    print(f"Best epoch:             {best_epoch} (validation accuracy {best_accuracy:.3f})")
    print(f"Test accuracy:          {test['accuracy']:.3f}")
    print(f"Test balanced accuracy: {test['balanced_accuracy']:.3f}")

    # 5. Save results and the trained model
    run_info = {'model': model_name, 'train_percent': train_percent, 'seed': seed, 'n_epochs': len(history),
                'best_epoch': best_epoch, 'learning_rate': LEARNING_RATE, 'n_parameters': n_parameters}
    save_run_results(project_root/'results/tables', run_info, history, test)

    models_directory = project_root/'results/models'
    models_directory.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'metadata': metadata,
                'best_epoch': best_epoch, 'best_validation_accuracy': float(best_accuracy)}, model_file)
    print('Results saved to', project_root/'results')

if __name__ == "__main__":
    main()
