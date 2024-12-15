import base64
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Union

import openai

# Imports for InternVL
import torch
import torchvision.transforms as T
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Constants for InternVL
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


def load_images_in_batches(
    image_paths: List[Path], input_size: int = 448, batch_size: int = 16
):
    transform = build_transform(input_size=input_size)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            batch_images.append(image)
        yield torch.stack(batch_images)  # Yield the batch tensor directly


def encode_image_openai(image_path: str) -> str:
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


def load_dataset(dataset_file: str):
    with open(dataset_file, "r") as f:
        return [json.loads(line) for line in f]


def run_inference(
    model_type: str,
    model: Union[OpenAI, PreTrainedModel],
    tokenizer: Union[str, PreTrainedTokenizer],
    question: str,
    image_paths: List[str],
    max_tokens: int = 10,
    history=None,
):
    """
    Runs inference given a question and images.

    model_type: 'openai' or 'internvl'
    model / tokenizer: Depending on model_type
    image_paths: paths to images
    question: the user's question
    history: used for multi-turn with internvl
    """

    if model_type == "openai":
        # Using OpenAI chat completion
        # Encode images as base64 data URLs
        image_blocks = []
        for img_file in image_paths:
            base64_image = encode_image_openai(img_file)
            image_blocks.append(
                {"type": "image_url", "image_url": {"url": base64_image}}
            )

        message_content = [{"type": "text", "text": question}] + image_blocks
        messages = [{"role": "user", "content": message_content}]

        response = model.chat.completions.create(
            model=tokenizer,  # tokenizer holds model_name in OpenAI scenario
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        model_answer_raw = response.choices[0].message.content.strip().lower()

    else:
        # InternVL
        # Load and preprocess images
        # Instead of directly chaining .to(), first collect the images:
        all_batches = []
        for batch in load_images_in_batches(
            [Path(p) for p in image_paths], input_size=448
        ):
            all_batches.append(batch)
        if len(all_batches) > 0:
            images = torch.cat(all_batches, dim=0).to(torch.bfloat16).cuda()
        else:
            # Handle the case if no images are provided
            images = None

        generation_config = {"max_new_tokens": max_tokens, "do_sample": False}
        # For multi-turn, we pass the history back and forth
        if history is not None:
            response, history = model.chat(
                tokenizer=tokenizer,
                pixel_values=images,
                question=question,
                generation_config=generation_config,
                history=history,
                return_history=True,
            )
        else:
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=images,
                question=question,
                generation_config=generation_config,
            )
            history = None

        model_answer_raw = response.strip().lower()

    # Normalize answer
    if "yes" in model_answer_raw:
        model_answer = "yes"
    elif "no" in model_answer_raw:
        model_answer = "no"
    else:
        # Default to "no" if we cannot find "yes"
        model_answer = "no"

    return model_answer, history


def benchmark_single_turn(
    dataset,
    model_type: str,
    model,
    tokenizer,
    image_paths: List[Path],
    data_path: str,
    max_tokens: int = 10,
    batch_size: int = 16,
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample_index, sample in enumerate(
        tqdm(dataset, desc="Benchmarking Single Turn")
    ):
        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()
        sample_image_paths = [
            Path(os.path.join(data_path, img_file)) for img_file in sample["image"]
        ]

        # Iterate through batches from the generator
        for batch_images in load_images_in_batches(
            sample_image_paths, input_size=448, batch_size=batch_size
        ):
            batch_images = batch_images.to(
                torch.bfloat16
            ).cuda()  # Move tensor batch to GPU and convert to bfloat16

            # Run inference for the batch
            model_answer, _ = run_inference(
                model_type=model_type,
                model=model,
                tokenizer=tokenizer,
                question=question,
                image_paths=[],  # No need for paths since we use batches
                max_tokens=max_tokens,
                history=None,
            )

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


def benchmark_multi_turn(
    dataset, model_type: str, model, tokenizer, data_path: str, max_tokens: int = 10
) -> Tuple[int, int, int, int, str]:
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""

    for sample_index, sample in enumerate(
        tqdm(dataset, desc="Benchmarking Multi Turn")
    ):
        image_files = [os.path.join(data_path, p) for p in sample["image"]]
        # For internVL multi-turn, we keep a history
        history = None
        model_answer = "no"  # default answer if none given yet

        for i, turn in enumerate(sample["conversations"]):
            if turn["from"] == "human":
                question = turn["value"]
                model_answer, history = run_inference(
                    model_type=model_type,
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    image_paths=image_files,
                    max_tokens=max_tokens,
                    history=history,
                )

                log_content += (
                    f"Sample {sample_index}, Turn {i}: User asked: {question}\n"
                )
                log_content += (
                    f"Sample {sample_index}, Turn {i}: Model answered: {model_answer}\n"
                )

            elif turn["from"] == "gpt":
                # This is the ground truth turn
                ground_truth = turn["value"].strip().lower()
                log_content += (
                    f"Sample {sample_index}, Turn {i}: Ground truth: {ground_truth}\n"
                )

                # Evaluate last model_answer
                if model_answer == "yes" and ground_truth == "yes":
                    tp += 1
                elif model_answer == "yes" and ground_truth == "no":
                    fp += 1
                elif model_answer == "no" and ground_truth == "no":
                    tn += 1
                elif model_answer == "no" and ground_truth == "yes":
                    fn += 1

    return tp, fp, tn, fn, log_content


def benchmark_combined(
    dataset, model_type: str, model, tokenizer, data_path: str, max_tokens: int = 10
) -> Tuple[int, int, int, int, str]:
    # The "combined" scenario: similar to single-turn,
    # but could be a special scenario.
    # We'll follow the logic from the internVL script's combined function.
    tp, fp, tn, fn = 0, 0, 0, 0
    log_content = ""
    for sample in tqdm(dataset, desc="Benchmarking Combined"):
        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()
        image_files = [
            os.path.join(data_path, img_file) for img_file in sample["image"]
        ]

        model_answer, _ = run_inference(
            model_type=model_type,
            model=model,
            tokenizer=tokenizer,
            question=question,
            image_paths=image_files,
            max_tokens=max_tokens,
        )

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


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    )
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    # Hardcode batch size
    BATCH_SIZE = 15000  # Adjust this value based on your system's resources

    parser = ArgumentParser(description="Run merged benchmarks for models")
    parser.add_argument("dataset_file", type=str, help="Path to the JSONL dataset file")
    parser.add_argument(
        "benchmark_type",
        type=str,
        choices=["single", "multi", "combined"],
        help="Whether to run single-turn, multi-turn or combined benchmark",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["openai", "internvl"],
        required=True,
        help="Type of model: openai for GPT-4o-mini, internvl for local InternVL model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="OpenAI model name if using openai type. Required if model_type is 'openai'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local model path if using internvl type. Required if model_type is 'internvl'.",
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

    # Validation: Combined benchmark is only valid for internvl
    if args.benchmark_type == "combined" and args.model_type != "internvl":
        parser.error(
            "The 'combined' benchmark type is only supported for model_type 'internvl'."
        )

    # Validation: Ensure required arguments are provided for specific model types
    if args.model_type == "openai" and not args.model_name:
        parser.error("You must specify --model_name when using model_type 'openai'.")
    if args.model_type == "internvl" and not args.model_path:
        parser.error("You must specify --model_path when using model_type 'internvl'.")

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

    result_logger = logging.getLogger("result_logger")
    result_handler = logging.FileHandler(args.result_file, mode="a")
    result_formatter = logging.Formatter("%(message)s")
    result_handler.setFormatter(result_formatter)
    result_logger.addHandler(result_handler)
    result_logger.setLevel(logging.INFO)

    logger.info(f"Model Type: {args.model_type}")
    if args.model_type == "openai":
        logger.info(f"Model Name: {args.model_name}")
    else:
        logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Dataset Path: {args.dataset_file}")
    logger.info("-------------------------------------")

    load_dotenv()

    if args.model_type == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        model = client  # We'll just pass client as model
        tokenizer = args.model_name  # We'll pass model_name as tokenizer arg
    else:
        model = (
            AutoModel.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .cuda()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

    dataset = load_dataset(args.dataset_file)
    data_path = str(Path(args.dataset_file).parent)
    image_paths = [
        Path(os.path.join(data_path, img_file))
        for sample in dataset
        for img_file in sample["image"]
    ]

    # Run benchmark
    if args.benchmark_type == "single":
        tp, fp, tn, fn, log_content = benchmark_single_turn(
            dataset=dataset,
            model_type=args.model_type,
            model=model,
            tokenizer=tokenizer,
            image_paths=image_paths,
            data_path=data_path,
            batch_size=BATCH_SIZE,  # Set a batch size suitable for your system
        )
    elif args.benchmark_type == "multi":
        tp, fp, tn, fn, log_content = benchmark_multi_turn(
            dataset=dataset,
            model_type=args.model_type,
            model=model,
            tokenizer=tokenizer,
            data_path=data_path,
        )
    else:  # combined
        tp, fp, tn, fn, log_content = benchmark_combined(
            dataset=dataset,
            model_type=args.model_type,
            model=model,
            tokenizer=tokenizer,
            data_path=data_path,
        )

    logger.info(log_content)

    # Compute metrics
    accuracy, precision, recall, f1_score = compute_metrics(tp, fp, tn, fn)

    results = (
        f"Model Type: {args.model_type}\n"
        f"{'Model Name: ' + args.model_name if args.model_type=='openai' else 'Model Path: ' + args.model_path}\n"
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
