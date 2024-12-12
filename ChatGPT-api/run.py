import base64
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import openai
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def load_dataset(dataset_file: str):
    with open(dataset_file, "r") as f:
        return [json.loads(line) for line in f]


def encode_image(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        mime_type = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    else:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def benchmark_gpt_single_turn(
    dataset, model_name: str, data_path: str, max_tokens: int = 10
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample in tqdm(dataset, desc="Benchmarking GPT - Single Turn"):
        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()

        # Encode images from local paths
        image_blocks = []
        for img_file in sample["image"]:
            img_full_path = os.path.join(data_path, img_file)
            base64_image = encode_image(img_full_path)
            image_blocks.append(
                {"type": "image_url", "image_url": {"url": base64_image}}
            )

        message_content = [{"type": "text", "text": question}] + image_blocks

        messages = [{"role": "user", "content": message_content}]

        response = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=max_tokens, temperature=0
        )

        model_answer_raw = response.choices[0].message.content.strip().lower()

        if "yes" in model_answer_raw:
            model_answer = "yes"
        elif "no" in model_answer_raw:
            model_answer = "no"
        else:
            model_answer = "no"

        log_content += f"Question: {question}\nModel Answer: {model_answer}\nGround Truth: {ground_truth}\n"

        if model_answer == "yes" and ground_truth == "yes":
            tp += 1
        elif model_answer == "yes" and ground_truth == "no":
            fp += 1
        elif model_answer == "no" and ground_truth == "no":
            tn += 1
        elif model_answer == "no" and ground_truth == "yes":
            fn += 1

    return tp, fp, tn, fn, log_content


def benchmark_gpt_multi_turn(
    dataset, model_name: str, data_path: str, max_tokens: int = 10
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample_index, sample in enumerate(
        tqdm(dataset, desc="Benchmarking GPT - Multi Turn")
    ):
        # Encode images once per sample
        image_descriptions = []
        for img_file in sample["image"]:
            image_descriptions.append(
                f"[Image attached: {img_file}]"
            )  # Describe images textually

        # Create a running list of messages for context
        messages = []

        turns = sample["conversations"]
        # Iterate over all conversation turns
        for i, turn in enumerate(turns):
            if turn["from"] == "human":
                # Construct the user's question
                user_question = f"{turn['value']}\n{' '.join(image_descriptions)}"
                messages.append({"role": "user", "content": user_question})

                # Call the model with the conversation history so far
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                )

                # Extract the model's response
                model_answer_raw = response.choices[0].message.content.strip().lower()
                if "yes" in model_answer_raw:
                    model_answer = "yes"
                elif "no" in model_answer_raw:
                    model_answer = "no"
                else:
                    model_answer = "no"

                # Log the turn
                log_content += (
                    f"Sample {sample_index}, Turn {i}: User asked: {user_question}\n"
                )
                log_content += (
                    f"Sample {sample_index}, Turn {i}: Model answered: {model_answer}\n"
                )

                # Add the assistant's response to the messages for context
                messages.append({"role": "assistant", "content": model_answer})

            elif turn["from"] == "gpt":
                # Ground truth is provided in the dataset
                ground_truth = turn["value"].strip().lower()
                log_content += (
                    f"Sample {sample_index}, Turn {i}: Ground truth: {ground_truth}\n"
                )

                # Evaluate the model's last response
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
    parser = ArgumentParser(description="Run benchmarks for GPT models")
    parser.add_argument("dataset_file", type=str, help="Path to the JSONL dataset file")
    parser.add_argument(
        "benchmark_type",
        type=str,
        choices=["single", "multi"],
        help="Whether to run single-turn or multi-turn benchmark",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (e.g. gpt-4o-mini)",
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

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)

    logging.basicConfig(
        filename=args.log_file,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    logger = logging.getLogger("benchmark_logger")

    # Results logger
    result_logger = logging.getLogger("result_logger")
    result_handler = logging.FileHandler(args.result_file, mode="a")
    result_formatter = logging.Formatter("%(message)s")
    result_handler.setFormatter(result_formatter)
    result_logger.addHandler(result_handler)
    result_logger.setLevel(logging.INFO)

    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Dataset Path: {args.dataset_file}")
    logger.info("-------------------------------------")

    dataset = load_dataset(args.dataset_file)
    data_path = str(Path(args.dataset_file).parent)

    if args.benchmark_type == "single":
        tp, fp, tn, fn, log_content = benchmark_gpt_single_turn(
            dataset=dataset,
            model_name=args.model_name,
            data_path=data_path,
        )
    else:
        tp, fp, tn, fn, log_content = benchmark_gpt_multi_turn(
            dataset=dataset,
            model_name=args.model_name,
            data_path=data_path,
        )

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
        f"Model Name: {args.model_name}\n"
        f"Dataset Path: {args.dataset_file}\n"
        f"Benchmark Type: {args.benchmark_type}\n"
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

    result_logger.info(results)
    logger.info("Benchmark completed. Results:")
    logger.info(results)
