import hashlib
import json
import logging
import math
import shutil
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
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
    """
    Samples `n` frames uniformly from the list of `frames` for each subsequence.
    Divides the frames into `subsequences` parts and samples `n` frames from each part.
    Ensures the last frame is always included in the last subsequence.
    Returns a list of subsequences, where each subsequence is a list of frame paths.
    """
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
    sampling_fn: Callable,
    frames_per_grid: int,
    subsequences_per_episode: int,
) -> list[np.ndarray]:
    """
    Creates image grids for each subsequence in an episode.
    Returns a list of np.ndarray grids.
    """

    grids = []

    for camera in CAMERAS:
        camera_frames = droid_episode[camera]
        subsequences_frames = sampling_fn(
            camera_frames, frames_per_grid, subsequences_per_episode
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


def save_images(images: list[np.ndarray], output_path: Path) -> list[Path]:
    image_paths = []
    for i, image in enumerate(images):
        image_path = output_path / f"{hashlib.md5(image).hexdigest()}.jpeg"
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

    def to_jsonl(self, with_id: int) -> dict:
        jsonl_entry = asdict(self)
        jsonl_entry.update({"id": with_id})
        return jsonl_entry


def generate_internvl_episodes(
    language_instruction: str,
    images: list[Path],
    label: bool,
) -> list[InternVLEpisode]:
    images_str = [p.as_posix() for p in images]
    height_list, width_list = zip(*(plt.imread(img).shape[:2] for img in images))

    episodes = []
    question = PROMPT_REASONING_GUIDANCE
    if len(images) > 1:
        question += "\n".join(f"Frame {j}: <image>" for j in range(len(images)))
    else:
        question += "<image>"
    question += f'\nHas the following task been achieved: "{language_instruction}"? Answer with "yes" or "no" only.'
    if width_list[0] != FRAME_WIDTH or height_list[0] != FRAME_HEIGHT:
        question = PROMPT_GRID_GUIDANCE + question

    episodes.append(
        InternVLEpisode(
            image=images_str,
            width_list=list(width_list),
            height_list=list(height_list),
            conversations=[
                {"from": "human", "value": question},
                {"role": "gpt", "value": "yes" if label else "no"},
            ],
        )
    )
    return episodes


def generate_internvl_dataset(
    dataset_path: Path,
    output_path: Path,
    frames_per_grid: int,
    multi_image: bool,
    subsequences_per_episode: int,
    process_fn: Callable = episode_to_image_grid,
    sampling_fn: Callable = sample_uniformly,
) -> None:
    start_read = perf_counter()
    dataset = pd.read_parquet(dataset_path / MANIFEST_FILE_NAME)
    end_read = perf_counter()
    logging.info(
        f"Read dataset of {len(dataset)} episodes in {end_read - start_read:.2f} seconds.\n"
    )
    logging.info(f"Dataset columns: {dataset.columns}\n")
    logging.info(f"Dataset types: {dataset.dtypes}\n")
    logging.info(f"First episode: {dataset.iloc[0]}\n")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Some language instructions are missing
    dataset["language_instructions"] = dataset["language_instructions"].apply(
        lambda x: [item for item in x if item != ""]
    )

    # Plot most frequent language instructions
    language_instructions = (
        dataset["language_instructions"]
        .explode()
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
    meta_file_path = output_path / "meta.jsonl"
    images_path = output_path / "images"
    images_path.mkdir(parents=True)

    num_internvl_episodes = 0
    for _, droid_episode in tqdm(dataset.iterrows(), total=len(dataset)):
        internvl_episodes = []
        language_instructions = droid_episode["language_instructions"]
        for language_instruction in language_instructions:
            images = process_fn(
                droid_episode=droid_episode,
                sampling_fn=sampling_fn,
                frames_per_grid=frames_per_grid,
                subsequences_per_episode=subsequences_per_episode,
            )
            image_paths = save_images(images=images, output_path=images_path)
            if multi_image:
                internvl_episodes.extend(
                    generate_internvl_episodes(
                        language_instruction=language_instruction,
                        images=image_paths,
                        label=True,
                    )
                )
            else:
                for i, image_path in enumerate(image_paths):
                    internvl_episodes.extend(
                        generate_internvl_episodes(
                            language_instruction=language_instruction,
                            images=[image_path],
                            label=i == len(image_paths) - 1,
                        )
                    )

        with open(meta_file_path, "a") as f:
            for internvl_episode in internvl_episodes:
                f.write(
                    json.dumps(internvl_episode.to_jsonl(with_id=num_internvl_episodes))
                    + "\n"
                )
                num_internvl_episodes += 1
        exit()


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
        "--subsequences_per_episode",
        type=int,
        default=1,
        help="Number of subsequences per episode.",
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
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    generate_internvl_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        frames_per_grid=args.frames_per_grid,
        multi_image=args.multi_image,
        subsequences_per_episode=args.subsequences_per_episode,
    )
