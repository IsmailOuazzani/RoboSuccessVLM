import json
from pathlib import Path
import sys
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform

def load_image(image_path, input_size=448):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_dataset(dataset_file):
    with open(dataset_file, "r") as f:
        return [json.loads(line) for line in f]

def benchmark_model(model, tokenizer, dataset, input_size=448):
    tp, fp, tn, fn = 0, 0, 0, 0

    for sample in tqdm(dataset, desc="Benchmarking Samples"):
        # Load multiple images if available
        image_paths = [Path(data_path) / "droid_3_1_1_multi_turn_48/data/" / p for p in sample["image"]]
        images = [load_image(ip, input_size=input_size) for ip in image_paths]
        images = torch.cat(images, dim=0).to(torch.bfloat16).cuda()

        # Each sample has multiple turns: human->gpt->human->gpt...
        # We'll track the conversation using 'history'
        history = None
        model_answer = None
        ground_truth = None

        # The pattern in your dataset: 
        # conversations: [ {from: human, value:...}, {from: gpt, value:...}, {from: human, value:...}, {from: gpt, value:...}, ... ]
        # The last gpt value is presumably the ground truth for the last question.
        
        # We will simulate the conversation turn-by-turn.
        # For each human turn, call model.chat() to get the model response.
        # For the subsequent gpt turn in the dataset, that is your ground truth, compare it with the last model response.

        for i, turn in enumerate(sample["conversations"]):
            if turn["from"] == "human":
                # This is a question to the model
                question = turn["value"]
                generation_config = {"max_new_tokens": 10, "do_sample": False}
                response, history = model.chat(
                    tokenizer,
                    images,
                    question,
                    generation_config,
                    history=history,
                    return_history=True,
                )
                model_answer = response.strip().lower()

                # Print statements to confirm multi-turn behavior
                print(f"Turn {i}: User asked: {question}")
                print(f"Turn {i}: Model answered: {response}")
                print(f"Turn {i}: Current history: {history}")

            else:  # turn["from"] == "gpt"
                # Ground truth for the previous user's question
                ground_truth = turn["value"].strip().lower()
                print(f"Turn {i}: Ground truth: {ground_truth}")

                # Evaluate performance
                if model_answer == "yes" and ground_truth == "yes":
                    tp += 1
                elif model_answer == "yes" and ground_truth == "no":
                    fp += 1
                elif model_answer == "no" and ground_truth == "no":
                    tn += 1
                elif model_answer == "no" and ground_truth == "yes":
                    fn += 1

        # Print metrics after each sample
        print(f"Metrics after sample {sample['id']} -> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    return tp, fp, tn, fn

if __name__ == "__main__":
    model_path = sys.argv[2]
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    data_path = sys.argv[1]

    dataset_file = f"{data_path}/droid_3_1_1_multi_turn_48/data/dataset.jsonl"
    dataset = load_dataset(dataset_file)

    tp, fp, tn, fn = benchmark_model(model, tokenizer, dataset)

    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
