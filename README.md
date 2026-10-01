# From artificial units to biologically informed networks

An exploratory NeuroAI project: does changing *how a single artificial neuron computes*, inspired by real neurons, change what a network learns?

## Research question

We compare a standard CNN against copies in which only the hidden-layer neurons are changed according to one biological idea at a time. Everything else — data split, training and evaluation — is kept identical.

We ask whether each biological idea changes:
1. **performance** on the task
2. **learning with fewer labels**
3. **robustness** to noisy inputs
4. the **internal representations** the network learns

We do not claim to model real neurons faithfully. Background and references are in [`docs/background.md`](docs/background.md).

## Biological context

A standard artificial neuron computes a weighted sum of all its inputs and applies one nonlinearity (e.g. ReLU). Real neurons differ in several ways. This project tests three of them, one at a time:

- **Dendritic branches:** in pyramidal neurons, dendritic branches process their own inputs nonlinearly before the soma combines them.
- **Dale's law:** a real neuron is either excitatory or inhibitory; all its outgoing synapses have the same sign.
- **Divisive normalization:** in visual cortex, a neuron's response is divided by the pooled activity of its neighbours.

## Dataset

**CIFAR-10**: 60,000 colour images (32×32) in 10 classes.

| Split | Images | Source |
|---|---:|---|
| Train | 45,000 | official training set |
| Validation | 5,000 | fixed random part of the official training set |
| Test | 10,000 | official test set, used only for the final evaluation |

- A small CNN does not reach ceiling performance on CIFAR-10, which leaves room to see whether a neuron-level change helps or hurts.
- The large test set makes small differences between models measurable.
- Link to neuroscience: object recognition is the task solved by the visual cortex, and CNNs were themselves inspired by it.

The dataset (~170 MB) is downloaded to `data/` the first time and is not included in the repository.

## Models

All models share the same CNN (two convolution blocks → hidden layer of 128 neurons → 10 outputs). Only the hidden-layer neurons differ:

| Model | Hidden neuron computation | Tests |
|---|---|---|
| `baseline` | `relu(w · x + c)` — standard artificial neuron | – |
| `linear_branches` | inputs split into 8 branches, `relu(Σ_b s_b (w_b · x_b + c_b) + d)` | control for `dendritic` |
| `dendritic` | same branches with their own nonlinearity, `relu(Σ_b s_b relu(w_b · x_b + c_b) + d)` | dendritic branches |
| `dale` | standard neuron; 80% of neurons excitatory (outgoing weights ≥ 0), 20% inhibitory (≤ 0) | Dale's law |
| `divisive_norm` | standard neuron, then `r_i = h_i / (σ + mean_j h_j)` with learned σ | divisive normalization |

- **Dendritic:** the 4,096 inputs of each neuron are assigned to 8 branches of 512 inputs using a fixed random order; every neuron still receives all inputs. `linear_branches` has exactly the same parameters as `dendritic` but is mathematically still a linear neuron, so comparing the two isolates the effect of the **branch nonlinearity**.
- **Dale:** the network learns the size of each outgoing weight; its sign is fixed by the neuron type.
- **Parameter counts** are saved in `summary.csv`: `baseline` and `dale` 545,098; `divisive_norm` +1 (σ); branch models +2,048 (+0.4%, branch and soma biases).
- With the same seed, all models start from the same convolution weights and see the training images in the same order.

## Experiments

| Experiment | What changes | Script |
|---|---|---|
| Main comparison | 5 models × 3 seeds, full training set | `train.py` |
| Fewer labels | 10%, 25%, 50%, 100% of the training images | `train.py` |
| Noise robustness | Gaussian noise added to the test images | `evaluate_noise.py` |
| Representations | hidden-layer activity on the test images | `analyze_representations.py` |

- **Fewer labels:** with fewer images the number of epochs is increased (e.g. 150 epochs for 10%), so every run makes the same number of weight updates. The validation and test sets never change.
- **Noise robustness:** noise standard deviation 0, 0.25, 0.5, 1.0 and 1.5 × the pixel standard deviation. The same noise is used for every model.
- **Representations:**
  - fraction of active neurons per image
  - fraction of neurons that are never active
  - class selectivity of neurons
  - similarity of the class-by-class representational geometry (RDM) to the baseline trained with the same seed

## Main parameters

| Parameter | Value |
|---|---:|
| Epochs (full training set) | 15 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Hidden neurons | 128 |
| Dendritic branches per neuron | 8 |
| Excitatory fraction (`dale`) | 0.8 |
| Seeds | 1, 2, 3 |

## Results

All numbers and figures are produced by the scripts and are never copied by hand. See `results/tables/` and `results/figures/`.

## Repository structure

```text
neuro-inspired-network/
├── LICENSE
├── README.md
├── data/                          # CIFAR-10, downloaded automatically (not in the repository)
├── docs/
│   └── background.md
├── requirements.txt
├── results/
│   ├── figures/
│   ├── models/                    # trained weights (not in the repository)
│   └── tables/
├── scripts/
│   ├── analyze_representations.py
│   ├── compare_models.py
│   ├── evaluate_noise.py
│   ├── make_figures.py
│   ├── run_all.sh
│   └── train.py
└── src/
    └── neuroinspired/
        ├── __init__.py
        ├── data.py
        ├── models.py
        ├── results.py
        └── training.py
```

## Running the analysis

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run everything (60 training runs, then analyses and figures):

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
python3 scripts/train.py dendritic 1          # model, seed (full training set)
python3 scripts/train.py dendritic 1 10       # model, seed, % of training images
python3 scripts/evaluate_noise.py
python3 scripts/analyze_representations.py
python3 scripts/compare_models.py
python3 scripts/make_figures.py
```

## Main outputs

```text
results/tables/summary.csv                     one row per training run
results/tables/model_comparison.csv            mean ± std over seeds, per model and training-set size
results/tables/noise_robustness.csv            one row per run and noise level
results/tables/noise_comparison.csv            mean ± std over seeds, per model and noise level
results/tables/representations.csv             hidden-layer measures, one row per run
results/tables/representation_comparison.csv   mean ± std over seeds, per model
results/tables/<run>_history.csv               training loss and validation accuracy per epoch
results/tables/<run>_confusion_matrix.csv      test confusion matrix (rows = true class)
results/tables/<run>_class_means.csv           mean hidden-layer activity per class
results/figures/                               figures made from the tables above
```

`<run>` is `<model>_train<percent>_seed<seed>`, e.g. `dendritic_train100_seed1`.

## License

The original code in this repository is available under the [MIT License](LICENSE). The CIFAR-10 dataset and the publications cited in this project remain subject to their respective original licenses and terms of use.
