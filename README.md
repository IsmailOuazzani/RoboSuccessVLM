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
Download the dataset (first 100 elements):
```
gsutil -m cp -r gs://gresearch/robotics/droid_100 <your_local_path>
```
or download the full dataset (1.7Tb):
```
gsutil -m cp -r gs://gresearch/robotics/droid <your_local_path>
```

Then run the pre-processing script with the relevant options:
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
