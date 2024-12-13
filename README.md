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
pre-processing.py --help
```

### Run the benchmark
InternVL documentation: https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html

### Fine tuning
#### Set up the environment
We use GPUs rented from [Lambda Labs](https://lambdalabs.com/) to fine-tune [InternVL](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html) models. Within a Lambda Labs instance, run the following to set up the fine tuning environment:
```
git clone git@github.com:IsmailOuazzani/CSC413-project.git
sh CSC413-project/scripts/setup-lambdalabs-env.sh
```
Alternatively, you could upload the `setup-lambdalabs-env.sh` file to your instance then run it.

#### Set up the dataset
Upload your dataset to the instance. After doing so, unzip and move the dataset to the correct directory. For example:
```
unzip droid_3_1_1_single_turn_432.zip
mv droid_3_1_1_single_turn_432.json InternVL/internvl_chat/shell/data/
mv data InternVL/internvl_chat/shell/data/
```

#### Set up the fine tuning script
We provide a fine tuning script in this repo, which is intended to be used for InternVL and uses LoRA. Move it to the correct location:
```
mv CSC413-project/scripts/finetune.sh InternVL/internvl_chat/shell/internvl2.0/2nd_finetune/
```

#### Fine tune the model
Go to the internvl_chat folder:
```
cd ~/InternVL/internvl_chat
```
Then download the [pretrained model](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html#model-preparation) of your choosing. For example:
```
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir pretrained/InternVL2-1B
```

Finally, start the fine-tuning with our script. Note that you should match the number of GPUs to your setup:
```
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/finetune.sh
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


#### Package the model
After the model has been fine-tuned, the LoRa weights need to be merged back in the original pretrained InternVL model. Instructions are given [here](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#merging-lora-weights).

Finally copy the Python scripts from the original InternVL2-2B directory to the new merged model directory so that your model can be used as an `AutoModel`. Instructions are given [here](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#wrapping-into-automodel).
