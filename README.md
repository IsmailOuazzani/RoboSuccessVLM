# CSC413 Project

## Set up
Create a virtual environment and install dependencies:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install [pre-commit](https://pre-commit.com/) configuration:
```
pre-commit install
```


## Usage

Activate virtual environment:
```
source venv/bin/activate
```

### Process the dataset
```
usage: pre-processing.py [-h] [--dataset_dir DATASET_DIR] [--split_size SPLIT_SIZE]
                         [--last_step_shift LAST_STEP_SHIFT]
                         [--num_subsequences NUM_SUBSEQUENCES]
                         [--steps_per_subsequence STEPS_PER_SUBSEQUENCE]
                         [--output_dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
  --split_size SPLIT_SIZE
  --last_step_shift LAST_STEP_SHIFT
  --num_subsequences NUM_SUBSEQUENCES
  --steps_per_subsequence STEPS_PER_SUBSEQUENCE
  --output_dir OUTPUT_DIR
```
