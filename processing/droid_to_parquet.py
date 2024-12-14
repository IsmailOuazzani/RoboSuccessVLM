import logging
import os
import shutil
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from constants import CAMERAS, MANIFEST_FILE_NAME
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class Episode:
    id: int
    language_instructions: list[str]
    exterior_image_1_left: list[str]
    exterior_image_2_left: list[str]
    wrist_image_left: list[str]

    def to_pd_row(self) -> dict:
        return asdict(self)


def extract_droid_episode(
    droid_episode: dict, output_files_path: Path, episode_id: int
) -> Episode:
    steps = list(droid_episode["steps"])
    step0 = steps[0]
    language_instructions = [
        step0["language_instruction"].numpy().decode("utf-8"),
        step0["language_instruction_2"].numpy().decode("utf-8"),
        step0["language_instruction_3"].numpy().decode("utf-8"),
    ]

    camera_frames: dict[str, list[str]] = {camera: [] for camera in CAMERAS}
    buffered_images = []

    for i, step in enumerate(steps):
        observation = step["observation"]
        for camera in CAMERAS:
            raw_frame = observation[camera]
            frame_path = output_files_path / f"{episode_id}_{camera}_{i}.jpeg"
            encoded_jpeg = tf.image.encode_jpeg(raw_frame, quality=85)
            buffered_images.append((encoded_jpeg, frame_path))
            camera_frames[camera].append(frame_path.as_posix())

    for encoded_jpeg, frame_path in buffered_images:
        tf.io.write_file(frame_path.as_posix(), encoded_jpeg)

    return Episode(
        id=episode_id,
        language_instructions=language_instructions,
        exterior_image_1_left=camera_frames["exterior_image_1_left"],
        exterior_image_2_left=camera_frames["exterior_image_2_left"],
        wrist_image_left=camera_frames["wrist_image_left"],
    )


def process_droid_dataset(
    dataset_path: Path,
    dataset_name: str,
    output_path: Path,
    chunk_size: int,
    max_episodes: int,
    start_episode: int,
):
    full_dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=dataset_path,
        split="train",
        with_info=True,
    )

    logging.info(f"Dataset info: {dataset_info}")
    num_episodes = len(full_dataset)
    logging.info(f"Number of episodes: {num_episodes}")
    if start_episode >= num_episodes:
        raise ValueError("Start episode is beyond the total number of episodes.")

    if max_episodes:  # Adjust num_episodes based on max_episodes
        num_episodes = min(num_episodes, start_episode + max_episodes)

    num_chunks = (num_episodes - start_episode) // chunk_size + 1
    logging.info(f"Number of episodes to process: {num_episodes - start_episode}")
    logging.info(f"Number of dataset chunks: {num_chunks}")
    del full_dataset

    logging.info("Initializing output directory")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    manifest_path = output_path / MANIFEST_FILE_NAME
    output_files_path = output_path / "images"
    output_files_path.mkdir(parents=True)
    logging.info(f"Output directory: {output_path}")

    num_episodes_extracted = 0
    episodes = []
    with tqdm(total=num_episodes - start_episode, desc="Processing Episodes") as pbar:
        for i in range(num_chunks):
            chunk_start = start_episode + i * chunk_size
            chunk_end = min(start_episode + (i + 1) * chunk_size, num_episodes)

            # Last minute hack
            if chunk_start == chunk_end:
                continue

            chunk_dataset = tfds.load(
                dataset_name,
                data_dir=dataset_path,
                # This dataset only has a train split
                split=f"train[{chunk_start}:{chunk_end}]",
            )
            for droid_episode in chunk_dataset:
                first_step = tf.data.experimental.get_single_element(
                    droid_episode["steps"].take(1)
                )
                language_instruction = (
                    first_step["language_instruction"].numpy().decode("utf-8")
                )
                if language_instruction:
                    episodes.append(
                        extract_droid_episode(
                            droid_episode=droid_episode,
                            output_files_path=output_files_path,
                            episode_id=num_episodes_extracted,
                        )
                    )
                    num_episodes_extracted += 1
                pbar.update(1)
    episodes_df = pd.DataFrame([episode.to_pd_row() for episode in episodes])
    episodes_df.to_parquet(manifest_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert the droid tensorflow dataset to parquet format."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path parent directory of the droid dataset.",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset directory."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the processed dataset."
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Maximum number of episodes to process. 0 means all episodes.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of episodes to process in a single chunk.",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Episode number to start processing from.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    dataset_name = args.dataset_name
    output_path = Path(args.output)
    logging.info(f"Args: {vars(args)}")
    process_droid_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        output_path=output_path,
        chunk_size=args.chunk_size,
        max_episodes=args.max_episodes,
        start_episode=args.start_episode,
    )
