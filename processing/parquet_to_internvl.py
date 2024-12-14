import json
import logging
import math
import shutil
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from constants import (
    CAMERAS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAMES_PER_IMAGE_GRID,
    MANIFEST_FILE_NAME,
    PROMPT_GRID_GUIDANCE,
    PROMPT_REASONING_GUIDANCE,
)
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def sample_uniformly(frames: list[str], n: int, subsequences: int) -> list[list[str]]:
    subsequences_list = []
    total_frames = len(frames)
    segment_length = total_frames // subsequences

    for i in range(subsequences):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < subsequences - 1 else total_frames
        segment = frames[start_idx:end_idx]
        indices = np.linspace(
            0,
            len(segment) - 1,
            n,
            dtype=int,
        ).tolist()
        if i == subsequences - 1 and len(segment) > 0:
            indices[-1] = (
                len(segment) - 1
            )  # Ensure the last frame is included in the last subsequence
        subsequences_list.append([segment[j] for j in indices])

    return subsequences_list


def episode_to_image_grid(
    droid_episode: pd.DataFrame,
    frames_per_grid: int,
    subsequences_per_episode: int,
) -> list[np.ndarray]:
    grids = []

    for camera in CAMERAS:
        camera_frames = droid_episode[camera]
        subsequences_frames = sample_uniformly(
            frames=camera_frames,
            n=frames_per_grid,
            subsequences=subsequences_per_episode,
        )

        for grid_frames in subsequences_frames:
            M = math.ceil(math.sqrt(frames_per_grid))
            rows, cols = (  # TODO: fix this logic to prefer horizontal grids
                (M, M - 1)
                if frames_per_grid == M * (M - 1)
                else (M, math.ceil(frames_per_grid / M))
            )
            grid = np.zeros(
                (rows * FRAME_HEIGHT, cols * FRAME_WIDTH, 3), dtype=np.uint8
            )
            for i, frame in enumerate(grid_frames):
                row, col = divmod(i, cols)
                grid[
                    row * FRAME_HEIGHT : (row + 1) * FRAME_HEIGHT,
                    col * FRAME_WIDTH : (col + 1) * FRAME_WIDTH,
                ] = plt.imread(frame)
            grids.append(grid)

    return grids


def save_images(
    images: list[np.ndarray], output_path: Path, start_index: int
) -> list[Path]:
    image_paths = []
    for i, image in enumerate(images):
        image_path = output_path / f"{start_index + i}.jpeg"
        if image_path.exists():
            raise RuntimeError(f"Image {image_path} already exists.")
        plt.imsave(image_path, image)
        image_paths.append(image_path)
    return image_paths


@dataclass
class InternVLEpisode:
    # https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#multi-image-data
    image: list[str]
    width_list: list[int]
    height_list: list[int]
    conversations: list[dict]
    success: bool

    def to_jsonl(self, with_id: int) -> dict:
        jsonl_entry = asdict(self)
        jsonl_entry.pop("success")
        jsonl_entry.update({"id": with_id})
        return jsonl_entry


def generate_internvl_episodes(
    language_instruction: str,
    negative_language_instructions: list[str],
    images: list[Path],
) -> list[InternVLEpisode]:
    images_str = [
        "images/" + p.name for p in images
    ]  # TODO: fix this to use relative paths
    height_list, width_list = zip(*(plt.imread(img).shape[:2] for img in images))

    episodes = []
    question = PROMPT_REASONING_GUIDANCE
    if len(images) > 1:
        for j in range(len(images)):
            camera_number = j // (len(images) // 3)
            question += f"Camera{camera_number} frame {j % (len(images) // 3)}: <image>"
    else:
        question += "<image>"
    question += "\nHas the following task been achieved:"
    if width_list[0] != FRAME_WIDTH or height_list[0] != FRAME_HEIGHT:
        question = PROMPT_GRID_GUIDANCE + question

    episodes.append(
        InternVLEpisode(
            image=images_str,
            width_list=list(width_list),
            height_list=list(height_list),
            conversations=[
                {"from": "human", "value": question + language_instruction},
                {"from": "gpt", "value": "yes"},
            ],
            success=True,
        )
    )
    for neg_language_instruction in negative_language_instructions:
        episodes.append(
            InternVLEpisode(
                image=images_str,
                width_list=list(width_list),
                height_list=list(height_list),
                conversations=[
                    {"from": "human", "value": question + neg_language_instruction},
                    {"from": "gpt", "value": "no"},
                ],
                success=False,
            )
        )
    return episodes


def label_instructions_by_id(dataset: pd.DataFrame):
    G = nx.Graph()
    G.add_nodes_from(dataset.index)

    instr_map: dict[str, list[int]] = {}
    for i, instructions in dataset["language_instructions"].items():
        for instr in instructions:
            instr_map.setdefault(instr, []).append(i)
    for rows in instr_map.values():
        for r in rows[1:]:
            G.add_edge(rows[0], r)

    components = list(nx.connected_components(G))
    id_map = {node: cid for cid, comp in enumerate(components, 1) for node in comp}
    dataset["instruction_id"] = dataset.index.map(id_map)


def generate_internvl_dataset(
    dataset: pd.DataFrame,
    output_path: Path,
    frames_per_grid: int,
    multi_image: bool,
    subsequences_per_episode: int,
    negative_ratio: int,
) -> list[InternVLEpisode]:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    dataset["language_instructions"] = dataset["language_instructions"].apply(
        lambda x: [item for item in x if item != ""]
    )
    label_instructions_by_id(dataset)
    dataset = (
        dataset.explode("language_instructions")
        .reset_index(drop=True)
        .rename(columns={"language_instructions": "language_instruction"})
    )

    # Plot most frequent language instructions
    (
        dataset["language_instruction"]
        .value_counts()
        .head(20)
        .sort_values()
        .plot(kind="barh", figsize=(10, 8))
    )
    plt.xlabel("Frequency")
    plt.ylabel("Language Instruction")
    plt.title("Top Language Instructions")
    plt.tight_layout()
    plt.savefig(output_path / "language_instruction_frequency.png")

    # As per https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html#prepare-your-customized-training-data
    meta_file_path = output_path / "meta.json"
    annotation_file_path = output_path / "annotation.jsonl"
    images_path = output_path / "images"
    images_path.mkdir(parents=True)

    num_internvl_episodes = 0
    image_count = 0
    internvl_episodes = []
    for _, droid_episode in tqdm(dataset.iterrows(), total=len(dataset)):
        language_instruction = droid_episode["language_instruction"]
        images = episode_to_image_grid(
            droid_episode=droid_episode,
            frames_per_grid=frames_per_grid,
            subsequences_per_episode=subsequences_per_episode,
        )
        image_paths = save_images(
            images=images, output_path=images_path, start_index=image_count
        )
        image_count += len(image_paths)
        negative_language_instructions = (
            dataset[dataset["instruction_id"] != droid_episode["instruction_id"]][
                "language_instruction"
            ]
            .sample(negative_ratio)
            .tolist()
        )
        if multi_image:
            internvl_episodes.extend(
                generate_internvl_episodes(
                    language_instruction=language_instruction,
                    images=image_paths,
                    negative_language_instructions=negative_language_instructions,
                )
            )
        else:
            for i, image_path in enumerate(image_paths):
                internvl_episodes.extend(
                    generate_internvl_episodes(
                        language_instruction=language_instruction,
                        images=[image_path],
                        negative_language_instructions=negative_language_instructions,
                    )
                )

    with open(annotation_file_path, "a") as f:
        for internvl_episode in internvl_episodes:
            f.write(
                json.dumps(internvl_episode.to_jsonl(with_id=num_internvl_episodes))
                + "\n"
            )
            num_internvl_episodes += 1

    meta_file_path.write_text(
        # https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#meta-file
        json.dumps(
            {
                "test": {
                    "root": output_path.name,
                    "annotation": output_path.name + "/annotation.jsonl",
                    "data_augment": False,
                    "repeat_time": 1,
                    "length": num_internvl_episodes,
                }
            }
        )
    )
    return internvl_episodes


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert parquet droid dataset to format compatible with InternVL fine-tuning."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the parquet dataset."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the InternVL compatible dataset.",
    )
    parser.add_argument(
        "--frames_per_grid",
        type=int,
        default=FRAMES_PER_IMAGE_GRID,
        help="Number of frames per grid.",
    )
    parser.add_argument(
        "--multi_image",
        action="store_true",
        help="Whether to use multiple images per episode.",
    )
    parser.add_argument(
        "--subsequences_per_episode",
        type=int,
        default=1,
        help="Number of subsequences per episode.",
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=1.0,
        help="Ratio of negative examples to positive examples.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    start_time = perf_counter()
    dataset = pd.read_parquet(dataset_path / MANIFEST_FILE_NAME)
    logging.info(
        f"Loaded {len(dataset)} episodes in {perf_counter() - start_time:.2f} seconds."
    )
    logging.info(f"Dataset columns: {dataset.columns}\n")
    logging.info(f"Dataset types: {dataset.dtypes}\n")
    logging.info(f"First episode: {dataset.iloc[0]}\n")

    internvl_episodes = generate_internvl_dataset(
        dataset=dataset,
        output_path=output_path,
        frames_per_grid=args.frames_per_grid,
        multi_image=args.multi_image,
        subsequences_per_episode=args.subsequences_per_episode,
        negative_ratio=args.negative_ratio,
    )
    logging.info(f"Generated {len(internvl_episodes)} InternVL episodes.")

    num_negatives = len([ep for ep in internvl_episodes if not ep.success])
    num_positives = len(internvl_episodes) - num_negatives
    (output_path / "README").write_text(
        json.dumps(vars(args))
        + f"\nNum Positives: {num_positives}\nNum Negatives: {num_negatives}"
    )
