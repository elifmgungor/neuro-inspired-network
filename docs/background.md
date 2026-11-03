# Biological background

The models in this project borrow a few ideas from neuroscience and apply them to artificial hidden neurons. Each captures one aspect of neural computation; none is a detailed model of a biological circuit. Implementations are in [models.py](../src/neuroinspired/models.py), with the experiments and results described in the [README](../README.md).

## Visual cortex and CNNs

Neurons in early visual cortex respond to local regions of an image. The hierarchy of simple and complex cells described by Hubel & Wiesel helped motivate architectures that combine local feature detection with pooling, including Fukushima's neocognitron and later CNNs.

Here, a small CNN provides the same feature extractor for each hidden-layer variant. CIFAR-10 provides an object recognition task with enough difficulty to compare the models.

- Hubel DH, Wiesel TN. *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.* J Physiol. 1962;160(1):106–154.
- Fukushima K. *Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.* Biol Cybern. 1980;36(4):193–202.
- Krizhevsky A. *Learning multiple layers of features from tiny images.* University of Toronto, 2009. [Dataset and report](https://www.cs.toronto.edu/~kriz/cifar.html).

## Dendritic branches

A standard artificial neuron combines its inputs in one weighted sum before applying an activation. Dendrites can perform local nonlinear computations before their outputs reach the soma. Poirazi, Brannon & Mel showed how this behaviour can be represented as a small two-layer network within a pyramidal neuron.

The dense CNN branch models assign each hidden neuron's 4,096 inputs to eight branches of 512. In `dendritic`, each branch applies ReLU; the soma then combines branch outputs with learned weights and applies another ReLU. In `linear_branches`, the branch ReLU is removed, while the soma ReLU remains. Both models have the same connections and parameter count, so their comparison tests the contribution of the branch nonlinearity.

The sparse variants give each branch 16 randomly sampled inputs. All input assignments are fixed during training. Random sampling is a simplification: biological branches can receive inputs with spatial or functional structure.

Chavlis & Poirazi studied dendritic MLPs in which each dendrite receives a small subset of raw pixels and connects to one soma. Their models achieved similar or higher accuracy than fully connected networks with far fewer parameters. They attributed the gains to restricted sampling and structured connectivity.

The MLP experiment here follows their random-sampling architecture and adds a version with linear dendrites. This comparison keeps connectivity fixed while testing the branch activation. The training schedule uses early stopping, which differs from the paper.

- Poirazi P, Brannon T, Mel BW. *Pyramidal neuron as two-layer neural network.* Neuron. 2003;37(6):989–999.
- Chavlis S, Poirazi P. *Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning.* Nat Commun. 2025. [doi](https://doi.org/10.1038/s41467-025-56297-9).

## Dale's law

Dale's principle motivates separating excitatory and inhibitory units rather than allowing a single artificial neuron to have both positive and negative outgoing weights. The `dale` model uses 102 excitatory and 26 inhibitory hidden neurons, approximately an 80/20 split.

`DaleLinear` fixes each neuron's outgoing weight signs and learns their magnitudes. Only the hidden-to-output connections are constrained; the model does not include a separate inhibitory circuit.

- Eccles JC, Fatt P, Koketsu K. *Cholinergic and inhibitory synapses in a pathway from motor-axon collaterals to motoneurones.* J Physiol. 1954;126(3):524–562.
- Cornford J, Kalajdzievski D, Leite M, Lamarquette A, Kullmann DM, Richards B. *Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units.* ICLR. 2021.

## Divisive normalization

Divisive normalization scales a neuron's response by pooled activity. In visual cortex, it helps describe how responses depend on the surrounding stimulus and its contrast.

`DivisiveNormalization` uses the following rule after the hidden-layer ReLU:

```text
r_i = h_i / (σ + mean_j h_j)
```

The offset σ is learned and kept positive. The pool is the whole hidden layer with equal weights, rather than a local neighbourhood.

- Carandini M, Heeger DJ. *Normalization as a canonical neural computation.* Nat Rev Neurosci. 2012;13(1):51–62.

## Representation analysis

Class selectivity compares a unit's mean activity for its preferred class with its mean activity for the other classes:

```text
selectivity = (preferred_mean − other_mean) / (preferred_mean + other_mean)
```

The analysis also measures positive activity. `dead_fraction` means “fraction never positive on the evaluation set”, not a fraction of nonfunctional neurons. The MLPs use LeakyReLU, so negative outputs can still carry information; activity and selectivity measures use only the positive part of the outputs. RDMs use the original activations.

A representational dissimilarity matrix (RDM) measures differences between mean activation patterns for pairs of classes. Comparing RDMs tests whether two models organize classes similarly without matching individual neurons.

- Morcos AS, Barrett DGT, Rabinowitz NC, Botvinick M. *On the importance of single directions for generalization.* ICLR. 2018.
- Kriegeskorte N, Mur M, Bandettini P. *Representational similarity analysis — connecting the branches of systems neuroscience.* Front Syst Neurosci. 2008;2:4.

The full bibliography is in [references.bib](references.bib).
