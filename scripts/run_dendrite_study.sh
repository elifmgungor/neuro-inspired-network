#!/bin/bash
# Part 2: where does the dendritic effect come from?
# Restricted sampling vs branch nonlinearity, in the CNN and in the MLP of Chavlis & Poirazi (2025).
# Run from the project root:
#     bash scripts/run_dendrite_study.sh
# Only checkpoints with matching provenance are skipped.
set -e

SEEDS="1 2 3"

train() {
    python3 scripts/train.py "$1" "$2" "$3" --skip-existing
}

# 1. CNN: 2 x 2 design (linear_branches and dendritic were trained in Part 1)
for seed in $SEEDS; do
    for train_percent in 100 10; do
        for model in linear_branches dendritic linear_branches_sparse dendritic_sparse; do
            train $model $seed $train_percent
        done
    done
done

# 2. MLP: replication of Chavlis & Poirazi (2025) plus the linear-dendrite control
for seed in $SEEDS; do
    for model in mlp_vanilla mlp_dendritic_sparse mlp_linear_branches_sparse; do
        train $model $seed 100
    done
done

# 3. Analyses, tables and figures
python3 scripts/evaluate_noise.py
python3 scripts/analyze_representations.py
python3 scripts/compare_models.py
python3 scripts/make_figures.py
