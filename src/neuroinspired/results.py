'''
Save the results of each training run as CSV files.
Every run writes its own tables, and one shared table (summary.csv)
collects the final test results of all runs, so numbers never
have to be copied by hand.
'''
import pandas as pd

from .data import CLASSES

def get_run_name(model_name, train_percent, seed):
    '''
    Name used for all files of one run, e.g. dendritic_train100_seed1
    '''
    return f'{model_name}_train{train_percent}_seed{seed}'

def update_table(table_file, new_rows, key_columns):
    '''
    Add rows to a CSV table. Existing rows with the same key_columns are replaced,
    so running an experiment again never creates duplicate rows.
    '''
    new_rows = pd.DataFrame(new_rows)
    if table_file.exists():
        table = pd.read_csv(table_file)
        is_old = table[key_columns].apply(tuple, axis=1).isin(new_rows[key_columns].apply(tuple, axis=1))
        table = pd.concat([table[~is_old], new_rows])
    else:
        table = new_rows
    table.to_csv(table_file, index=False)

def save_run_results(tables_directory, run_info, history, test):
    '''
    Save the results of one training run.
    Parameters:
        - tables_directory: Folder where the CSV files are written (results/tables).
        - run_info: Dictionary describing the run (model, train_percent, seed, n_epochs, best_epoch, learning_rate, n_parameters).
        - history: List of dictionaries with the training loss and validation accuracy of each epoch.
        - test: Output of evaluate() on the test set.
    '''
    tables_directory.mkdir(parents=True, exist_ok=True)
    run_name = get_run_name(run_info['model'], run_info['train_percent'], run_info['seed'])

    # 1. Learning curve of this run
    pd.DataFrame(history).to_csv(tables_directory/f'{run_name}_history.csv', index=False)

    # 2. Confusion matrix of this run (rows = true class, columns = predicted class)
    confusion = pd.DataFrame(test['confusion_matrix'], index=CLASSES, columns=CLASSES)
    confusion.to_csv(tables_directory/f'{run_name}_confusion_matrix.csv')

    # 3. One row in the shared summary table
    row = dict(run_info)
    row['test_accuracy'] = round(test['accuracy'], 4)
    row['test_balanced_accuracy'] = round(test['balanced_accuracy'], 4)
    update_table(tables_directory/'summary.csv', [row], ['model', 'train_percent', 'seed'])
