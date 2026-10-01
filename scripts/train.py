'''
Train and evaluate one model on CIFAR-10.
Results are written to results/tables and the trained model to results/models.

Usage:
    python3 scripts/train.py <model> <seed> [train_percent]

Examples:
    python3 scripts/train.py baseline 1
    python3 scripts/train.py dendritic 1 10    (only 10% of the training images)
'''
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root/'src'))
from neuroinspired.data import load_cifar10
from neuroinspired.models import CNN
from neuroinspired.training import train_one_epoch, evaluate
from neuroinspired.results import get_run_name, save_run_results

N_EPOCHS = 15  # epochs on the full training set
LEARNING_RATE = 1e-3

def main():
    model_name = sys.argv[1]
    seed = int(sys.argv[2])
    train_percent = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    run_name = get_run_name(model_name, train_percent, seed)

    # With less data, train for more epochs so every run makes the same number of weight updates
    n_epochs = N_EPOCHS * 100 // train_percent

    torch.manual_seed(seed)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # 1. Data
    train_loader, validation_loader, test_loader = load_cifar10(project_root/'data', train_percent, random_state=seed)

    # 2. Model and optimizer
    model = CNN(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'Run: {run_name}, device: {device}, parameters: {n_parameters:,}, training images: {len(train_loader.dataset):,}')

    # Reset the seed so every model sees the training images in the same order
    torch.manual_seed(seed)

    # 3. Training
    history = []
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        history.append({'epoch': epoch, 'train_loss': round(train_loss, 4), 'validation_accuracy': round(validation['accuracy'], 4)})
        print(f"Epoch {epoch:3d} | train loss {train_loss:.3f} | validation accuracy {validation['accuracy']:.3f}")

    # 4. Final evaluation on the test set
    test = evaluate(model, test_loader, device)
    print()
    print(f"Test accuracy:          {test['accuracy']:.3f}")
    print(f"Test balanced accuracy: {test['balanced_accuracy']:.3f}")

    # 5. Save results and the trained model
    run_info = {'model': model_name, 'train_percent': train_percent, 'seed': seed, 'n_epochs': n_epochs,
                'learning_rate': LEARNING_RATE, 'n_parameters': n_parameters}
    save_run_results(project_root/'results/tables', run_info, history, test)

    models_directory = project_root/'results/models'
    models_directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_directory/f'{run_name}.pt')
    print('Results saved to', project_root/'results')

if __name__ == "__main__":
    main()
