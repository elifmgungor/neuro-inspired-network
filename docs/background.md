# Background and references

## Dataset

Krizhevsky A. **Learning multiple layers of features from tiny images.** Technical report, University of Toronto, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html

## Why CNNs are linked to the visual cortex

Hubel DH, Wiesel TN. **Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.** *J Physiol*. 1962;160(1):106–154.

- Neurons in early visual cortex respond to small, local parts of the image (receptive fields).
- "Simple" and "complex" cells form a hierarchy.

Fukushima K. **Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.** *Biol Cybern*. 1980;36(4):193–202.

- First network built on this hierarchy; convolution + pooling in modern CNNs follow the same idea.

## Neuron-level change 1: dendritic branches

Poirazi P, Brannon T, Mel BW. **Pyramidal neuron as two-layer neural network.** *Neuron*. 2003;37(6):989–999.

- A standard artificial neuron sums all inputs at once and applies one nonlinearity.
- In real pyramidal neurons, dendritic branches first apply their own nonlinearity to their inputs; the soma then combines the branch outputs.
- This can be modelled as a small two-layer network inside a single neuron.

In this project (`DendriticLayer` in `src/neuroinspired/models.py`):

- Each hidden neuron has 8 branches; each branch receives 512 of the 4,096 inputs (fixed random assignment).
- `dendritic`: each branch applies ReLU, the soma takes a weighted sum of the branches and applies ReLU again.
- `linear_branches`: identical, but without the branch ReLU. This control has the same parameters and is still a linear neuron, so any difference from `dendritic` comes from the branch nonlinearity.
- Simplification: real branches receive inputs from specific, often related sources; here the assignment is random.

## Neuron-level change 2: Dale's law

Eccles JC, Fatt P, Koketsu K. **Cholinergic and inhibitory synapses in a pathway from motor-axon collaterals to motoneurones.** *J Physiol*. 1954;126(3):524–562.

Cornford J, Kalajdzievski D, Leite M, Lamarquette A, Kullmann DM, Richards B. **Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units.** *ICLR*. 2021.

- A neuron releases the same neurotransmitter at all its synapses, so it is either excitatory or inhibitory ("Dale's principle", named by Eccles).
- In standard artificial networks, one neuron can have positive and negative outgoing weights at the same time.
- In cortex, roughly 80% of neurons are excitatory and 20% inhibitory.

In this project (`DaleLinear` in `src/neuroinspired/models.py`):

- 102 of the 128 hidden neurons are excitatory (all outgoing weights ≥ 0), 26 are inhibitory (all ≤ 0).
- The network learns the size of each weight; the sign is fixed.
- Simplification: only the outgoing weights of the hidden layer follow the rule; there is no separate inhibitory circuit.

## Neuron-level change 3: divisive normalization

Carandini M, Heeger DJ. **Normalization as a canonical neural computation.** *Nat Rev Neurosci*. 2012;13(1):51–62.

- In visual cortex (and many other areas), a neuron's response is divided by the summed activity of a pool of neighbouring neurons.
- This makes responses depend on relative rather than absolute input strength (e.g. contrast).

In this project (`DivisiveNormalization` in `src/neuroinspired/models.py`):

- `r_i = h_i / (σ + mean_j h_j)`, where `h` is the ReLU output of the hidden layer and the pool is the whole layer.
- σ is learned and kept positive.
- Simplification: the pool is every hidden neuron with equal weight, not a local neighbourhood.

## Analysis methods

Morcos AS, Barrett DGT, Rabinowitz NC, Botvinick M. **On the importance of single directions for generalization.** *ICLR*. 2018.

- Class selectivity index of a neuron: (mean activity for the preferred class − mean for the other classes) / (sum of both).

Kriegeskorte N, Mur M, Bandettini P. **Representational similarity analysis — connecting the branches of systems neuroscience.** *Front Syst Neurosci*. 2008;2:4.

- Representational dissimilarity matrix (RDM): how different the average activity patterns for every pair of classes are. Comparing RDMs compares the "geometry" of two representations without matching individual neurons.
