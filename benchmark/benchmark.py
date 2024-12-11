import json
from pathlib import Path
import sys
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import os

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

def load_images(image_paths, input_size=448):
    transform = build_transform(input_size=input_size)
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)
    images = torch.stack(images)  # Stack images into a batch
    return images

def load_dataset(dataset_file):
    with open(dataset_file, "r") as f:
        return [json.loads(line) for line in f]

def save_log(log_path, content, model_path, dataset_path, config_info):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Check if the file exists to decide whether to write the header
    file_exists = os.path.isfile(log_path)

    # Write the header if the file does not exist
    with open(log_path, "a") as f:
        if not file_exists:
            header = (
                f"Model Path: {model_path}\n"
                f"Dataset Path: {dataset_path}\n"
                f"Config Info: {config_info}\n"
                "-------------------------------------\n"
            )
            f.write(header)
        f.write(content + "\n")  # Append log content

def save_results(result_path, tp, fp, tn, fn, model_path, dataset_path, config_info):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    )

    results = (
        f"Model Path: {model_path}\n"
        f"Dataset Path: {dataset_path}\n"
        f"Config Info: {config_info}\n"
        "-------------------------------------\n"
        f"True Positives: {tp}\n"
        f"False Positives: {fp}\n"
        f"True Negatives: {tn}\n"
        f"False Negatives: {fn}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1_score:.4f}\n"
        "-------------------------------------\n"
    )

    # Append the results (including header) to the file
    with open(result_path, "a") as f:
        f.write(results)
    print(results)


def benchmark_single_turn(model, tokenizer, dataset, input_size, data_path):
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample in tqdm(dataset, desc="Benchmarking Single Turn"):
        image_path = Path(data_path) / sample["image"][0]
        image = load_images([image_path], input_size=input_size).to(torch.bfloat16).cuda()

        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()

        generation_config = {"max_new_tokens": 10, "do_sample": False}
        response = model.chat(tokenizer, image, question, generation_config)

        log_content += f"Question: {question}\n"
        log_content += f"Model Answer: {response}\n"
        log_content += f"Ground Truth: {ground_truth}\n"

        model_answer = response.strip().lower()
        if model_answer == "yes" and ground_truth == "yes":
            tp += 1
        elif model_answer == "yes" and ground_truth == "no":
            fp += 1
        elif model_answer == "no" and ground_truth == "no":
            tn += 1
        elif model_answer == "no" and ground_truth == "yes":
            fn += 1

    return tp, fp, tn, fn, log_content

def benchmark_multi_turn(model, tokenizer, dataset, input_size, data_path):
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample in tqdm(dataset, desc="Benchmarking Multi Turn"):
        image_paths = [Path(data_path) / p for p in sample["image"]]
        images = load_images(image_paths, input_size=input_size).to(torch.bfloat16).cuda()

        history = None

        for i, turn in enumerate(sample["conversations"]):
            if turn["from"] == "human":
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
                log_content += f"Turn {i}: User asked: {question}\n"
                log_content += f"Turn {i}: Model answered: {response}\n"

            else:
                ground_truth = turn["value"].strip().lower()
                log_content += f"Turn {i}: Ground truth: {ground_truth}\n"
                if response.strip().lower() == "yes" and ground_truth == "yes":
                    tp += 1
                elif response.strip().lower() == "yes" and ground_truth == "no":
                    fp += 1
                elif response.strip().lower() == "no" and ground_truth == "no":
                    tn += 1
                elif response.strip().lower() == "no" and ground_truth == "yes":
                    fn += 1

    return tp, fp, tn, fn, log_content

def benchmark_combined(model, tokenizer, dataset, input_size, data_path):
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample in tqdm(dataset, desc="Benchmarking Combined"):
        image_paths = [Path(data_path) / p for p in sample["image"]]
        images = load_images(image_paths, input_size=input_size).to(torch.bfloat16).cuda()

        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()

        generation_config = {"max_new_tokens": 10, "do_sample": False}
        response = model.chat(tokenizer, images, question, generation_config)

        log_content += f"Question: {question}\n"
        log_content += f"Model Answer: {response}\n"
        log_content += f"Ground Truth: {ground_truth}\n"

        model_answer = response.strip().lower()
        if model_answer == "yes" and ground_truth == "yes":
            tp += 1
        elif model_answer == "yes" and ground_truth == "no":
            fp += 1
        elif model_answer == "no" and ground_truth == "no":
            tn += 1
        elif model_answer == "no" and ground_truth == "yes":
            fn += 1

    return tp, fp, tn, fn, log_content

if __name__ == "__main__":
    dataset_file = sys.argv[1]  # Full path to the JSONL file
    model_path = sys.argv[2]  # Model path or identifier
    benchmark_type = sys.argv[3]  # "single", "multi", or "combined"
    log_file = "result/log.txt"
    result_file = "result/result.txt"

    # Example config info (can include additional details as needed)
    config_info = f"Benchmark Type: {benchmark_type}, Input Size: 448"

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

    dataset = load_dataset(dataset_file)
    data_path = str(Path(dataset_file).parent)

    if benchmark_type == "single":
        tp, fp, tn, fn, log_content = benchmark_single_turn(
            model, tokenizer, dataset, 448, data_path
        )
    elif benchmark_type == "multi":
        tp, fp, tn, fn, log_content = benchmark_multi_turn(
            model, tokenizer, dataset, 448, data_path
        )
    elif benchmark_type == "combined":
        tp, fp, tn, fn, log_content = benchmark_combined(
            model, tokenizer, dataset, 448, data_path
        )
    else:
        raise ValueError("Unknown benchmark type. Use 'single', 'multi', or 'combined'.")

    save_log(log_file, log_content, model_path, dataset_file, config_info)
    save_results(result_file, tp, fp, tn, fn, model_path, dataset_file, config_info)
