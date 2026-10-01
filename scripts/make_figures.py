'''
Make figures from the saved result tables.
Figures are only drawn from results/tables, so every figure
can be traced back to a CSV file.
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIRECTORY = PROJECT_ROOT/'results/tables'
FIGURES_DIRECTORY = PROJECT_ROOT/'results/figures'

# Fixed color and marker for every model, so a model looks the same in every figure
MODEL_STYLES = {
    'baseline':        {'color': '#2a78d6', 'marker': 'o'},
    'linear_branches': {'color': '#eb6834', 'marker': 's'},
    'dendritic':       {'color': '#1baf7a', 'marker': '^'},
    'dale':            {'color': '#eda100', 'marker': 'D'},
    'divisive_norm':   {'color': '#e87ba4', 'marker': 'v'},
}

def style_axis(ax):
    '''
    Light grid behind the data and no top/right frame lines.
    '''
    ax.grid(color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

def save(fig, file_name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIRECTORY/file_name, dpi=300)
    plt.close(fig)

def first_seed_run_name(summary, model):
    '''
    Run name of the first seed of a model trained on the full training set.
    '''
    seed = summary[(summary['model'] == model) & (summary['train_percent'] == 100)]['seed'].min()
    return f'{model}_train100_seed{seed}'

def plot_learning_curves(summary):
    '''
    Validation accuracy per epoch (full training set), averaged over seeds.
    '''
    fig, ax = plt.subplots(figsize=(7, 4.5))
    runs = summary[summary['train_percent'] == 100]
    for model, style in MODEL_STYLES.items():
        seeds = runs[runs['model'] == model]['seed']
        if len(seeds) == 0:
            continue
        histories = [pd.read_csv(TABLES_DIRECTORY/f'{model}_train100_seed{seed}_history.csv') for seed in seeds]
        mean_accuracy = pd.concat(histories).groupby('epoch')['validation_accuracy'].mean()
        ax.plot(mean_accuracy.index, mean_accuracy.values, color=style['color'], linewidth=2, label=model)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy')
    ax.set_title('Learning curves (mean over seeds)')
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, 'learning_curves.png')

def plot_test_accuracy(summary):
    '''
    Test accuracy of every model (full training set): each seed as a small dot,
    mean ± standard deviation as a large marker with error bar.
    '''
    fig, ax = plt.subplots(figsize=(7, 4.5))
    runs = summary[summary['train_percent'] == 100]
    models = [m for m in MODEL_STYLES if m in set(runs['model'])]

    for i, model in enumerate(models):
        accuracy = runs[runs['model'] == model]['test_accuracy']
        style = MODEL_STYLES[model]
        ax.scatter(np.full(len(accuracy), i + 0.15), accuracy, color=style['color'], s=15, alpha=0.6)
        ax.errorbar(i, accuracy.mean(), yerr=accuracy.std() if len(accuracy) > 1 else 0,
                    color=style['color'], marker=style['marker'], markersize=8, capsize=4, linewidth=2)

    ax.set_xticks(range(len(models)), models, rotation=20)
    ax.set_ylabel('Test accuracy')
    ax.set_title('Test accuracy (full training set)')
    style_axis(ax)
    save(fig, 'test_accuracy.png')

def plot_lines(table, x_column, y_column, x_label, title, file_name, log_x=False):
    '''
    One line per model with mean ± standard deviation over seeds.
    '''
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, style in MODEL_STYLES.items():
        rows = table[table['model'] == model].sort_values(x_column)
        if len(rows) == 0:
            continue
        ax.errorbar(rows[x_column], rows[f'{y_column}_mean'], yerr=rows[f'{y_column}_std'].fillna(0),
                    color=style['color'], marker=style['marker'], markersize=7, linewidth=2, capsize=3, label=model)
    if log_x:
        ax.set_xscale('log')
        ax.set_xticks(sorted(table[x_column].unique()), [str(x) for x in sorted(table[x_column].unique())])
    ax.set_xlabel(x_label)
    ax.set_ylabel('Test accuracy')
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, file_name)

def plot_confusion_matrix(run_name):
    '''
    Test confusion matrix of one run as row percentages
    (each row shows how the images of one true class were classified).
    '''
    confusion = pd.read_csv(TABLES_DIRECTORY/f'{run_name}_confusion_matrix.csv', index_col=0)
    percentages = 100 * confusion.div(confusion.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(percentages, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(confusion.index)), confusion.index)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(f'Test confusion matrix (%): {run_name}')
    fig.colorbar(image)
    save(fig, f'{run_name}_confusion_matrix.png')

def plot_class_rdms(summary):
    '''
    Representational dissimilarity matrix (1 - correlation between class-average
    hidden-layer patterns) of every model, first seed.
    '''
    models = [m for m in MODEL_STYLES if (TABLES_DIRECTORY/f'{first_seed_run_name(summary, m)}_class_means.csv').exists()]
    if len(models) == 0:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        class_means = pd.read_csv(TABLES_DIRECTORY/f'{first_seed_run_name(summary, model)}_class_means.csv', index_col=0)
        image = ax.imshow(1 - np.corrcoef(class_means.values), cmap='Blues', vmin=0, vmax=2)
        ax.set_xticks(range(len(class_means)), class_means.index, rotation=90, fontsize=7)
        # Class names on the y axis only in the first panel, so they do not overlap the next panel
        ax.set_yticks(range(len(class_means)), class_means.index if model == models[0] else [], fontsize=7)
        ax.set_title(model)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label='1 - correlation')
    fig.savefig(FIGURES_DIRECTORY/'class_rdms.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(TABLES_DIRECTORY/'summary.csv')

    # 1. Main comparison (full training set)
    plot_learning_curves(summary)
    plot_test_accuracy(summary)
    for model in MODEL_STYLES:
        if model in set(summary['model']):
            plot_confusion_matrix(first_seed_run_name(summary, model))

    # 2. Learning with fewer labels
    comparison = pd.read_csv(TABLES_DIRECTORY/'model_comparison.csv')
    if comparison['train_percent'].nunique() > 1:
        plot_lines(comparison, 'train_percent', 'test_accuracy', 'Training images used (%)',
                   'Learning with fewer labels', 'low_data.png', log_x=True)

    # 3. Robustness to noise
    if (TABLES_DIRECTORY/'noise_comparison.csv').exists():
        noise = pd.read_csv(TABLES_DIRECTORY/'noise_comparison.csv')
        plot_lines(noise, 'noise_std', 'test_accuracy', 'Noise standard deviation (× pixel std)',
                   'Robustness to input noise', 'noise_robustness.png')

    # 4. Hidden-layer representations
    plot_class_rdms(summary)

    print('Figures saved to', FIGURES_DIRECTORY)

if __name__ == "__main__":
    main()
