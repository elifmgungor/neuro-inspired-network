# From artificial units to biologically informed networks

An exploratory NeuroAI project: does changing *how a single artificial neuron computes*, inspired by real neurons, change what a network learns?

## Research question

**Where does the benefit of dendrites in artificial networks come from — from restricted input sampling or from the nonlinearity inside each branch — and does it hold inside a CNN and with few training labels?**

The project started with a broader question: we compared a standard CNN against copies in which only the hidden-layer neurons were changed according to one biological idea at a time, and asked whether each idea changes
1. **performance** on the task
2. **learning with fewer labels**
3. **robustness** to noisy inputs
4. the **internal representations** the network learns

The results of that first experiment, together with the literature, led to the focused question above (see [How the question developed](#how-the-question-developed)).

We do not claim to model real neurons faithfully. More background on each mechanism is in [`docs/background.md`](docs/background.md).

## How the question developed

1. **Three biological ideas from the literature.** A standard artificial neuron computes a weighted sum of all its inputs and applies one nonlinearity. Real neurons differ, and we implemented three well-known differences as changes to the hidden layer of a small CNN:
   - **Dendritic branches:** Poirazi, Brannon & Mel (2003) showed that a pyramidal neuron behaves like a small two-layer network, because each dendritic branch processes its own inputs nonlinearly before the soma combines them.
   - **Dale's law:** a neuron is either excitatory or inhibitory (Eccles, Fatt & Koketsu, 1954); Cornford et al. (2021) showed how artificial networks can follow this rule.
   - **Divisive normalization:** in visual cortex a neuron's response is divided by the pooled activity of its neighbours (Carandini & Heeger, 2012).

   To isolate the dendritic nonlinearity we added a control, `linear_branches`, with exactly the same branches but without the nonlinearity.

2. **A first, unexpected result.** On CIFAR-10 (Krizhevsky, 2009) the dendritic neuron performed *worse* than the standard neuron for every seed and every training-set size, and the gap grew as the number of training images decreased. The linear-branch control did not show this drop, so it came from the branch nonlinearity (`results/tables/paired_comparison.csv`). Dale's law and divisive normalization did not clearly change accuracy; divisive normalization did make the hidden layer sparser and more class-selective (`results/tables/representation_comparison.csv`).

3. **Reading the literature.** This looked like the opposite of Chavlis & Poirazi (2025), who found that dendritic artificial networks learn more accurately, overfit less and need far fewer parameters. Reading their paper closely, we noticed three differences from our setup:
   - their networks are multilayer perceptrons on raw pixels, not CNNs;
   - each of their dendrites receives only **16 inputs** (restricted sampling), and they attribute the gains to this restricted sampling and the structured connectivity;
   - we did not find a control with **linear dendrites**, nor experiments that vary the **number of training images**.

   They also report that dendritic networks use mixed-selective rather than class-specific units, which matches the lower class selectivity we measured for our dendritic model.

4. **Ideas we considered and set aside.** Combining dendrites, excitatory/inhibitory units and divisive (shunting) inhibition has recently been studied by Safaai et al. (2026); deep networks under Dale's constraint by Abel & Ullman (2026); divisive normalization in CNNs on CIFAR-10 by Cekic et al. (2022); and circuits inspired by the fly mushroom body by Dasgupta, Stevens & Navlakha (2017) and Shen, Dasgupta & Navlakha (2023). We therefore did not make these the focus.

5. **The focused question.** Separating restricted sampling from the branch nonlinearity, in a CNN and in the original MLP setting, and with fewer training labels, is the part we did not find in the literature. Our literature search was limited, so this needs to be checked further before any claim of novelty.

## Dataset

**CIFAR-10** (Krizhevsky, 2009): 60,000 colour images (32×32) in 10 classes.

| Split | Images | Source |
|---|---:|---|
| Train | 45,000 | official training set |
| Validation | 5,000 | fixed random part of the official training set |
| Test | 10,000 | official test set, used only for the final evaluation |

- A small CNN does not reach ceiling performance on CIFAR-10, which leaves room to see whether a neuron-level change helps or hurts.
- The large test set makes small differences between models measurable.
- Link to neuroscience: object recognition is the task solved by the visual cortex (Hubel & Wiesel, 1962), and CNNs were themselves inspired by it (Fukushima, 1980).

The dataset (~170 MB) is downloaded to `data/` the first time and is not included in the repository.

## Models

### CNN models

All CNN models share the same convolution layers (two convolution blocks → hidden layer of 128 neurons → 10 outputs). Only the hidden-layer neurons differ:

| Model | Hidden neuron computation | Inputs per branch |
|---|---|---|
| `baseline` | `relu(w · x + c)` — standard artificial neuron | – |
| `linear_branches` | 8 branches, `relu(Σ_b s_b (w_b · x_b + c_b) + d)` — control | 512 (all inputs split) |
| `dendritic` | 8 branches with their own ReLU, `relu(Σ_b s_b relu(w_b · x_b + c_b) + d)` | 512 (all inputs split) |
| `linear_branches_sparse` | like `linear_branches` | 16 random |
| `dendritic_sparse` | like `dendritic` | 16 random |
| `dale` | standard neuron; 80% of neurons excitatory (outgoing weights ≥ 0), 20% inhibitory (≤ 0) | – |
| `divisive_norm` | standard neuron, then `r_i = h_i / (σ + mean_j h_j)` with learned σ | – |

- **All inputs split:** the 4,096 inputs of each neuron are assigned to 8 branches of 512 using a fixed random order, so every neuron still receives all inputs.
- **16 random:** every branch of every neuron samples its own 16 inputs, as in Chavlis & Poirazi (2025). These models have far fewer parameters.
- With the same seed, all CNN models start from the same convolution weights and see the training images in the same order.

### MLP models (replication of Chavlis & Poirazi, 2025)

Networks on raw pixels with 4,096 first-layer units (dendrites) and 256 second-layer units (somata); LeakyReLU (0.1) everywhere, as in the paper. The paper trains for a fixed 50 epochs; we use the same early stopping as for the CNN models, because the fully connected network was still improving at epoch 50.

| Model | Description | In the paper |
|---|---|---|
| `mlp_vanilla` | fully connected 3072 → 4096 → 256 → 10 | vANN |
| `mlp_dendritic_sparse` | 256 somata × 16 dendrites, each dendrite samples 16 random pixels | dANN-R |
| `mlp_linear_branches_sparse` | same, but dendrites without nonlinearity | not tested |

Parameter counts of every model are saved in `results/tables/summary.csv`.

## Experiments

### Part 1: three biological ideas (completed)

`baseline`, `linear_branches`, `dendritic`, `dale`, `divisive_norm` × seeds 1–3 × 10%, 25%, 50%, 100% of the training images (`bash scripts/run_all.sh`).

### Part 2: where does the dendritic effect come from? (in progress)

A 2 × 2 design, in the CNN and in the MLP:

| | Linear branches | Nonlinear branches |
|---|---|---|
| **Branches see all inputs** | `linear_branches` | `dendritic` |
| **16 inputs per branch** | `linear_branches_sparse` | `dendritic_sparse` |

- CNN: seeds 1–3 × 10% and 100% of the training images.
- MLP: `mlp_vanilla`, `mlp_dendritic_sparse`, `mlp_linear_branches_sparse` × seeds 1–3 × 100%.
- Run with `bash scripts/run_dendrite_study.sh`.

Comparisons decided before running (`results/tables/paired_comparison.csv`):
1. Branch nonlinearity with restricted sampling: `dendritic_sparse` − `linear_branches_sparse`
2. Restricted sampling with nonlinear branches: `dendritic_sparse` − `dendritic`
3. Replication: `mlp_dendritic_sparse` − `mlp_vanilla` (Chavlis & Poirazi report a positive difference)
4. The missing control in the MLP: `mlp_dendritic_sparse` − `mlp_linear_branches_sparse`

Any other comparison is exploratory.

### Methods shared by all experiments

- **Model selection:** in every run, the weights of the epoch with the highest validation accuracy are evaluated on the test set (`best_epoch` in `summary.csv`). The test set is never used for this choice.
- **Early stopping:** every model is trained until the validation accuracy has not improved for 10 epochs, with at most 100 epochs (on the full training set). The actual number of epochs is `n_epochs` in `summary.csv`.
- **Fewer labels:** an epoch on fewer images has fewer weight updates, so the maximum and the patience are scaled up (e.g. 10% of the images: at most 1,000 epochs, stop after 100 epochs without improvement). The validation and test sets never change.
- **How the training length was chosen:** the first runs used a fixed 15 epochs (CNN), then 30. In both cases some runs, especially the `*_sparse` models, reached their best validation accuracy in the last epochs, so we switched to early stopping. The earlier results are kept in the git history. Because training is deterministic, a run gives exactly the same result under the old and new rules unless its validation accuracy improved again more than 10 epochs after an earlier best.
- **Paired comparisons:** models trained with the same seed share the data split, data order and (for CNNs) initial convolution weights, so they are compared seed by seed.
- **Noise robustness:** Gaussian noise of 0, 0.25, 0.5, 1.0 and 1.5 × the pixel standard deviation added to the test images; the same noise for every model.
- **Representations:** fraction of active units, fraction of units never active, class selectivity (Morcos et al., 2018), and similarity of the class-by-class representational geometry to the reference model with the same seed (representational similarity analysis, Kriegeskorte, Mur & Bandettini, 2008).

## Main parameters

| Parameter | Value |
|---|---:|
| Early stopping, full training set | patience 10 epochs, at most 100 epochs |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| CNN hidden neurons / branches per neuron | 128 / 8 |
| MLP somata / dendrites per soma / inputs per dendrite | 256 / 16 / 16 |
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
│   ├── background.md              # the biological mechanisms in more detail
│   └── references.bib             # all references (BibTeX)
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
│   ├── run_all.sh                 # Part 1
│   ├── run_dendrite_study.sh      # Part 2
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

Run the experiments (finished runs are skipped, so the scripts can be stopped and restarted):

```bash
caffeinate -i bash scripts/run_all.sh              # Part 1
caffeinate -i bash scripts/run_dendrite_study.sh   # Part 2
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
results/tables/paired_differences.csv          model − reference, per training-set size and seed
results/tables/paired_comparison.csv           paired differences summarized over seeds
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

## References

All references are also in [`docs/references.bib`](docs/references.bib). Papers are cited, not included in the repository; follow the links to read them.

**Dataset and visual cortex**
- Krizhevsky A. *Learning multiple layers of features from tiny images.* Technical report, University of Toronto, 2009. [link](https://www.cs.toronto.edu/~kriz/cifar.html)
- Hubel DH, Wiesel TN. Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *J Physiol* 1962;160(1):106–154. [doi](https://doi.org/10.1113/jphysiol.1962.sp006837)
- Fukushima K. Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. *Biol Cybern* 1980;36(4):193–202. [doi](https://doi.org/10.1007/BF00344251)

**Dendrites**
- Poirazi P, Brannon T, Mel BW. Pyramidal neuron as two-layer neural network. *Neuron* 2003;37(6):989–999. [doi](https://doi.org/10.1016/S0896-6273(03)00149-1)
- Chavlis S, Poirazi P. Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning. *Nat Commun* 2025. [doi](https://doi.org/10.1038/s41467-025-56297-9) · [arXiv](https://arxiv.org/abs/2404.03708)
- Safaai H, Richards M, Khoshnevis N, Sabatini BL. When branch-local shunting helps: a gain-load-alignment principle for dendritic E/I networks. 2026. [arXiv](https://arxiv.org/abs/2607.24990)

**Dale's law**
- Eccles JC, Fatt P, Koketsu K. Cholinergic and inhibitory synapses in a pathway from motor-axon collaterals to motoneurones. *J Physiol* 1954;126(3):524–562. [doi](https://doi.org/10.1113/jphysiol.1954.sp005226)
- Cornford J, Kalajdzievski D, Leite M, Lamarquette A, Kullmann DM, Richards B. Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units. *ICLR* 2021.
- Abel R, Ullman S. Learning in deep networks under Dale's constraint. 2026. [arXiv](https://arxiv.org/abs/2608.06963)

**Divisive normalization**
- Carandini M, Heeger DJ. Normalization as a canonical neural computation. *Nat Rev Neurosci* 2012;13(1):51–62. [doi](https://doi.org/10.1038/nrn3136)
- Cekic M, et al. Neuro-inspired deep neural networks with sparse, strong activations. 2022. [arXiv](https://arxiv.org/abs/2202.13074)

**Analysis methods**
- Morcos AS, Barrett DGT, Rabinowitz NC, Botvinick M. On the importance of single directions for generalization. *ICLR* 2018. [arXiv](https://arxiv.org/abs/1803.06959)
- Kriegeskorte N, Mur M, Bandettini P. Representational similarity analysis — connecting the branches of systems neuroscience. *Front Syst Neurosci* 2008;2:4. [doi](https://doi.org/10.3389/neuro.06.004.2008)

**Considered but not pursued**
- Dasgupta S, Stevens CF, Navlakha S. A neural algorithm for a fundamental computing problem. *Science* 2017;358(6364):793–796. [doi](https://doi.org/10.1126/science.aam9868)
- Shen Y, Dasgupta S, Navlakha S. Reducing catastrophic forgetting with associative learning: a lesson from fruit flies. 2023. [pdf](https://cseweb.ucsd.edu/~dasgupta/papers/SDN23.pdf)

## License

The original code in this repository is available under the [MIT License](LICENSE). The CIFAR-10 dataset and the publications cited in this project remain subject to their respective original licenses and terms of use.
