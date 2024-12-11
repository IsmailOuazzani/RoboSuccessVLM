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

#### Running the Benchmark
1. **Ensure the directory structure is correct:**
```
project_directory/
├── data/                 # Contains unzipped │   
├── datasets such as droid_3_3_1_multi_turn_48
│   ├── ...
├── benchmark/
│   ├── benchmark331Single.py
│   ├── benchmark331Multiple.py
│   ├── benchmark331Combined.py
│   ├── benchmark311Single.py
│   ├── benchmark311Multiple.py
│   ├── benchmark311Combined.py
│   ├── result/                   
│   │   ├── log.txt
│   │   ├── result.txt
│   ├── run.py
├──pre-processing.py
├──...
```

2. **Run the `run.py` script:**
Before running, please update the correct model path in run.py, the default is "OpenGVLab/InternVL2-1B".
From the `benchmark` folder, execute:
```
python run.py
```


#### Package the model
After the model has been fine-tuned, the LoRa weights need to be merged back in the original pretrained InternVL model. Instructions are given [here](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#merging-lora-weights).

Finally copy the Python scripts from the original InternVL2-2B directory to the new merged model directory so that your model can be used as an `AutoModel`. Instructions are given [here](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#wrapping-into-automodel).
