import ast
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
from droid_to_parquet import MANIFEST_FILE_NAME
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FRAME_WIDTH = 320
FRAME_HEIGHT = 180

FRAMES_PER_IMAGE_GRID = 6


@dataclass
class InternVLEpisode:
    id: int
    image: list[str]
    width_list: list[int]
    height_list: list[int]
    conversations: list[dict]

    def to_jsonl(self) -> dict:
        return asdict(self)


def sample_uniformly(frames: list[str], n: int) -> list[str]:
    indices = np.linspace(
        0,
        len(frames) - 1,
        n,
        dtype=int,
    ).tolist()
    return [frames[j] for j in indices]


def episode_to_image_grid(
    episode: pd.DataFrame, sampling_fn: Callable, images_path: Path, output_path: Path
) -> InternVLEpisode:
    wrist_frames = ast.literal_eval(
        episode["wrist_image_left"]
    )  # TODO: figure out why this is a string
    grid_frames = sampling_fn(wrist_frames, FRAMES_PER_IMAGE_GRID)
    M = math.ceil(math.sqrt(FRAMES_PER_IMAGE_GRID))
    rows, cols = (
        (M, M - 1)
        if FRAMES_PER_IMAGE_GRID == M * (M - 1)
        else (M, math.ceil(FRAMES_PER_IMAGE_GRID / M))
    )
    grid = np.zeros((rows * FRAME_HEIGHT, cols * FRAME_WIDTH, 3), dtype=np.uint8)
    for i, frame in enumerate(grid_frames):
        row, col = divmod(i, cols)
        grid[
            row * FRAME_HEIGHT : (row + 1) * FRAME_HEIGHT,
            col * FRAME_WIDTH : (col + 1) * FRAME_WIDTH,
        ] = plt.imread(frame)
    image_path = images_path / f"{episode['id']}.jpeg"
    plt.imsave(images_path / f"{episode['id']}.jpeg", grid)

    internvlEpisode = InternVLEpisode(
        id=episode["id"],
        image=[str(image_path.relative_to(output_path))],
        width_list=[FRAME_WIDTH],
        height_list=[FRAME_HEIGHT],
        conversations=[
            {
                "from": "human",
                "value": f"<image>\nHas the following task been achieved:\"{episode['language_instruction_1']}\"? Answer with \"yes\" or \"no\" only.",
            },
            {
                "from": "gpt",
                "value": "yes" if True else "no",
            },
        ],
    )
    return internvlEpisode


def generate_internvl_dataset(
    dataset_path: Path,
    output_path: Path,
    process_fn: Callable = episode_to_image_grid,
    sampling_fn: Callable = sample_uniformly,
) -> None:
    start_read = perf_counter()
    dataset = pd.read_parquet(dataset_path / MANIFEST_FILE_NAME)
    end_read = perf_counter()
    logging.info(f"Read dataset in {end_read - start_read:.2f} seconds.")
    logging.info(f"Loaded dataset with {len(dataset)} episodes.")
    logging.info(f"Dataset columns: {dataset.columns}")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Plot the top language instructions
    language_instructions = dataset["language_instruction_1"].value_counts().head(20)
    language_instructions.sort_values().plot(kind="barh", figsize=(10, 8))
    plt.xlabel("Frequency")
    plt.ylabel("Language Instruction")
    plt.title("Top Language Instructions")
    plt.tight_layout()
    plt.savefig(output_path / "language_instruction_frequency.png")

    meta_file_path = output_path / "meta.jsonl"
    images_path = output_path / "images"
    images_path.mkdir(parents=True)

    for _, episode in tqdm(dataset.iterrows(), total=len(dataset)):
        internvl_episode = process_fn(
            episode=episode,
            sampling_fn=sampling_fn,
            images_path=images_path,
            output_path=output_path,
        )
        with open(meta_file_path, "a") as f:
            f.write(json.dumps(internvl_episode.to_jsonl()) + "\n")


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
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    generate_internvl_dataset(dataset_path=dataset_path, output_path=output_path)
