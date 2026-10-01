#!/bin/bash
# Run the complete experiment: every model, 3 seeds, 4 training-set sizes,
# then the analyses and figures. Run from the project root:
#     bash scripts/run_all.sh
set -e

MODELS="baseline linear_branches dendritic dale divisive_norm"
SEEDS="1 2 3"
TRAIN_PERCENTS="100 50 25 10"

# 1. Training (runs whose trained model already exists are skipped,
#    so the script can be stopped and started again)
for seed in $SEEDS; do
    for train_percent in $TRAIN_PERCENTS; do
        for model in $MODELS; do
            if [ -f "results/models/${model}_train${train_percent}_seed${seed}.pt" ]; then
                echo "Skipping ${model}_train${train_percent}_seed${seed} (already done)"
            else
                python3 scripts/train.py $model $seed $train_percent
            fi
        done
    done
done

# 2. Analyses (use the models trained on the full training set)
python3 scripts/evaluate_noise.py
python3 scripts/analyze_representations.py

# 3. Tables across seeds and figures
python3 scripts/compare_models.py
python3 scripts/make_figures.py
