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

Run `processing/droid_to_parquet.py` to convert the droid dataset (tensorflow format) into an interim more usable format. It is recommended to set a maximum number of episodes. For example
```
python processing/droid_to_parquet.py --dataset_path /media/ismail/WDC --dataset_name droid --split train --max_episodes 25 --output interim-dataset-example
```

Then, convert the dataset to a format compatible with InternVL fine tuning:
```
# For example, for the multi image format:
python processing/parquet_to_internvl.py --dataset_path out --output_path .idea/dffinal --frames_per_grid 1 --subsequences_per_episode 6
# For the image grid format:
python processing/parquet_to_internvl.py --dataset_path out --output_path .idea/dffinal --frames_per_grid 6
```

### Run the benchmark
InternVL documentation: https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html

### Fine tuning
#### Set up the Lambda Labs environment
We use GPUs rented from [Lambda Labs](https://lambdalabs.com/) to fine-tune [InternVL](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html) models. Within a Lambda Labs instance, run the following to set up the fine tuning environment.

First, [set up Docker](https://docs.lambdalabs.com/education/programming/virtual-environments-containers/) on the instance:
```
sudo adduser "$(id -un)" docker
sudo usermod -aG docker $USER
newgrp docker
```

#### Set up the dataset
Upload your dataset to the instance and unzip it:
```
unzip droid_3_1_1_single_turn_432.zip
```

#### Fine tuning scripts
Upload your fine tuning script to the instance. Choose one from the `./scripts` folder in this repository.
- `internvl_1b_imagegrid_11.sh` Use to train the 1B InternVL model on the imagegrid dataset with a 1:1 positive negative ratio.
- - `internvl_1b_multiimage_11.sh` Use to train the 1B InternVL model on the multi_image dataset with a 1:1 positive negative ratio.

#### Fine tune the model
Launch the fine tuning container:
```
newgrp docker
docker run --gpus all -it --name finetuning  ismailoz/internvl:latest
```

From the host machine, in a different shell, copy the relevant files (dataset and fine tuning script) to the container:
```
newgrp docker
docker cp . finetuning:/workspace/
```

Finally, within the container, start the fine-tuning with our script. Note that you should match the number of GPUs to your setup:
```
GPUS=1 PER_DEVICE_BATCH_SIZE=1 sh finetune.sh
```


### Benchmark Script Usage

#### Running `run.py` in ChatGPT-API or InternVL

The `run.py` script allows you to benchmark datasets using either the ChatGPT API (e.g., OpenAI models) or a local pretrained model (e.g., InternVL). You can process datasets in single-turn, multi-turn, or combined conversation modes.

---

#### For ChatGPT-API
Ensure you have an OpenAI API key. Set it as an environment variable:
  ```bash
  export OPENAI_API_KEY="your_openai_api_key"
  ```

#### Usage:
```
python run.py <dataset_path> <benchmark_type> --model_type <model_type> [OPTIONS]
```

#### Arguments:
- `<dataset_path>`: Full path to the dataset's JSONL file.
- `<benchmark_type>`: Type of benchmark to run. Options are:
  - `single`: Single-turn conversations.
  - `multi`: Multi-turn conversations.
  - `combined`: Combined single-turn conversation mode (InternVL only).
- `--model_type`: Type of model to use. Options are:
  - `openai`: Use the OpenAI ChatGPT API (e.g., GPT-4).
  - `internvl`: Use a locally stored pretrained model (e.g., InternVL).
- **Options for `--model_type`:**
  - `--model_name`: Specify the OpenAI model name (e.g., `gpt-4o-mini`). **Required for `openai`.**
  - `--model_path`: Path or identifier for the pretrained InternVL model (e.g., `OpenGVLab/InternVL2-1B`). **Required for `internvl`.**

#### Output:
- Logs will be saved in `/benchmark/result/log.txt`.
- Results will be saved in `/benchmark/result/result.txt`.

---

#### ChatGPT API (OpenAI Models) example
Single-turn benchmark:
```
python run.py /home/ubuntu/data/droid_3_3_1_single_turn_48/data/dataset.jsonl single --model_type openai --model_name gpt-4o-mini
```
#### InterVL example
Combined benchmark:
```
python run.py /home/ubuntu/data/droid_3_3_1_single_turn_combined_48/data/dataset.jsonl combined --model_type internvl --model_path OpenGVLab/InternVL2-1B
```
=======

#### Running the `run.py` script in `/benchmark/`:**

```
python run.py <dataset_path> <model_path> <benchmark_type>
```

- `<dataset_path>`: Full path to the dataset's JSONL file. For example: `/data/droid_3_1_1_single_turn_combined_48/data/dataset.jsonl`

- `<model_path>`: Path or identifier for the pre-trained model. For example: `OpenGVLab/InternVL2-1B`.
- `<benchmark_type>`: Type of benchmark to run (`single`, `multi`, or `combined`).
- The results will be saved in `/benchmark/result/log.txt` and `/benchmark/result/result.txt`



#### Package the model
After the model has been fine-tuned, the LoRa weights need to be merged back in the original pretrained InternVL model. Instructions are given [here](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#merging-lora-weights).

Finally copy the Python scripts from the original InternVL2-2B directory to the new merged model directory so that your model can be used as an `AutoModel`. Instructions are given [here](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#wrapping-into-automodel).
