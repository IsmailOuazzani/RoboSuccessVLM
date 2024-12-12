import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
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


def load_images(image_paths: List[Path], input_size: int = 448) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)
    images = torch.stack(images)  # Stack images into a batch
    return images


def load_dataset(dataset_file: str) -> List[dict]:
    with open(dataset_file, "r") as f:
        return [json.loads(line) for line in f]


def benchmark_single_turn(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    dataset: List[dict],
    input_size: int,
    data_path: str,
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""
    for sample in tqdm(dataset, desc="Benchmarking Single Turn"):
        image_path = Path(data_path) / sample["image"][0]
        image = (
            load_images([image_path], input_size=input_size).to(torch.bfloat16).cuda()
        )
        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()
        generation_config = {"max_new_tokens": 10, "do_sample": False}

        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=image,  # pixel_values replaces image
            question=question,
            generation_config=generation_config,
        )

        log_content += f"Question: {question}\nModel Answer: {response}\nGround Truth: {ground_truth}\n"
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


def benchmark_multi_turn(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    dataset: List[dict],
    input_size: int,
    data_path: str,
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""
    for sample in tqdm(dataset, desc="Benchmarking Multi Turn"):
        image_paths = [Path(data_path) / p for p in sample["image"]]
        images = (
            load_images(image_paths, input_size=input_size).to(torch.bfloat16).cuda()
        )
        history = None
        response = ""
        for i, turn in enumerate(sample["conversations"]):
            if turn["from"] == "human":
                question = turn["value"]
                generation_config = {"max_new_tokens": 10, "do_sample": False}
                response, history = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=images,
                    question=question,
                    generation_config=generation_config,
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


def benchmark_combined(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    dataset: List[dict],
    input_size: int,
    data_path: str,
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""
    for sample in tqdm(dataset, desc="Benchmarking Combined"):
        image_paths = [Path(data_path) / p for p in sample["image"]]
        images = (
            load_images(image_paths, input_size=input_size).to(torch.bfloat16).cuda()
        )
        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()
        generation_config = {"max_new_tokens": 10, "do_sample": False}
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=images,
            question=question,
            generation_config=generation_config,
        )
        log_content += f"Question: {question}\nModel Answer: {response}\nGround Truth: {ground_truth}\n"
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
    parser = ArgumentParser(description="Run benchmarks for InternVL models")
    parser.add_argument("dataset_file", type=str, help="Path to the JSONL dataset file")
    parser.add_argument("model_path", type=str, help="Path or identifier for the model")
    parser.add_argument(
        "benchmark_type",
        type=str,
        choices=["single", "multi", "combined"],
        help="Benchmark type",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="result/log.txt",
        help="Path to save the log file",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="result/result.txt",
        help="Path to save the result file",
    )
    args = parser.parse_args()

    # Ensure directories exist for logging
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    # Configure logging for logs
    logging.basicConfig(
        filename=args.log_file,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    logger = logging.getLogger("benchmark_logger")

    # Configure a separate logger for results
    result_logger = logging.getLogger("result_logger")
    result_handler = logging.FileHandler(args.result_file, mode="a")
    result_formatter = logging.Formatter("%(message)s")
    result_handler.setFormatter(result_formatter)
    result_logger.addHandler(result_handler)
    result_logger.setLevel(logging.INFO)

    config_info = f"Benchmark Type: {args.benchmark_type}, Input Size: 448"

    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Dataset Path: {args.dataset_file}")
    logger.info(f"Config Info: {config_info}")
    logger.info("-------------------------------------")

    model = (
        AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_dataset(args.dataset_file)
    data_path = str(Path(args.dataset_file).parent)

    if args.benchmark_type == "single":
        tp, fp, tn, fn, log_content = benchmark_single_turn(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            input_size=448,
            data_path=data_path,
        )
    elif args.benchmark_type == "multi":
        tp, fp, tn, fn, log_content = benchmark_multi_turn(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            input_size=448,
            data_path=data_path,
        )
    else:
        tp, fp, tn, fn, log_content = benchmark_combined(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            input_size=448,
            data_path=data_path,
        )

    # Log the details of each Q&A from log_content
    logger.info(log_content)

    # Compute metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    )

    results = (
        f"Model Path: {args.model_path}\n"
        f"Dataset Path: {args.dataset_file}\n"
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

    # Log results to the result file using result_logger
    result_logger.info(results)
    # Also log these metrics to the main logger
    logger.info("Benchmark completed. Results:")
    logger.info(results)
