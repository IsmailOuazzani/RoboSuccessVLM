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


def benchmark_model(model, tokenizer, dataset, input_size=448):
    tp, fp, tn, fn = 0, 0, 0, 0

    for sample in tqdm(dataset, desc="Benchmarking Samples"):
        image_paths = [Path(data_path) / "droid_3_3_1_single_turn_432/data/" / img_path for img_path in sample["image"]]
        images = load_images(image_paths, input_size=input_size).to(torch.bfloat16).cuda()

        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"].strip().lower()

        generation_config = {"max_new_tokens": 10, "do_sample": False}

        # Pass the list of images to the model
        response = model.chat(tokenizer, images, question, generation_config)

        print(f"Question: {question}")
        print(f"Model Answer: {response}")
        print(f"Ground Truth: {ground_truth}")

        model_answer = response.strip().lower()
        if model_answer == "yes" and ground_truth == "yes":
            tp += 1
        elif model_answer == "yes" and ground_truth == "no":
            fp += 1
        elif model_answer == "no" and ground_truth == "no":
            tn += 1
        elif model_answer == "no" and ground_truth == "yes":
            fn += 1

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
    dataset_file = f"{data_path}/droid_3_3_1_single_turn_432/data/dataset.jsonl"
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