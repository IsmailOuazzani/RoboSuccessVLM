#!/bin/bash

# Set variables
DATASET_PATH="/home/ubuntu/csc413/CSC413-project/data/imagegrid_11_neg_validation/annotation.jsonl"  # Path to your dataset
MODEL_PATH="/home/ubuntu/csc413/CSC413-project/finetuned/model_internvl2b_imagegrid_11"
BENCHMARK_TYPE="single"  # Options: single, multi, combined
MODEL_TYPE="internvl"    # Options: openai, internvl

# Run the benchmark script
python run.py $DATASET_PATH $BENCHMARK_TYPE --model_type $MODEL_TYPE --model_path $MODEL_PATH
