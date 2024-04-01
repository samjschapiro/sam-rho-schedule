#!/bin/bash

# Define the array of models to iterate over
models=("resnet18" "resnet50" "wide_resnet50_2")

# Define the array of datasets to iterate over
datasets=("cifar10" "cifar100")

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each dataset
    for dataset in "${datasets[@]}"; do
        echo "Running experiment with model ${model} on dataset ${dataset}..."
        # Call your Python script with the current model and dataset
        python3 main.py --model $model --dataset $dataset --epochs 1
        echo "Experiment with model ${model} on dataset ${dataset} completed."
    done
done

echo "All experiments completed."
