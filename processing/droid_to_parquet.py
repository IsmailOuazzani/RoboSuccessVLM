import logging
import os
import shutil
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import fastparquet
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MANIFEST_FILE_NAME = "manifest.parquet"
CAMERAS = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]


@dataclass
class Episode:
    id: int
    language_instruction_1: str
    language_instruction_2: str
    language_instruction_3: str
    exterior_image_1_left: list[str]
    exterior_image_2_left: list[str]
    wrist_image_left: list[str]

    def to_pd_row(self) -> dict:
        return asdict(self)


def extract_droid_episode(
    droid_episode: dict, output_files_path: Path, episode_id: int
) -> Episode:
    """
    Args:
        droid_episode (dict): Episode data from the droid dataset.
        output_files_path (Path): Directory to save the episode frames.
        episode_id (int): Episode id.

    Returns:
        Episode: Episode data.
    """
    steps = list(droid_episode["steps"])
    step0 = steps[0]
    language_instructions = [
        step0["language_instruction"].numpy().decode("utf-8"),
        step0["language_instruction_2"].numpy().decode("utf-8"),
        step0["language_instruction_3"].numpy().decode("utf-8"),
    ]

    camera_frames: dict[str, list[str]] = {camera: [] for camera in CAMERAS}
    for i, step in enumerate(steps):
        observation = step["observation"]
        for camera in CAMERAS:
            raw_frame = observation[camera]
            frame_path = output_files_path / f"{episode_id}_{camera}_{i}.jpeg"
            tf.io.write_file(frame_path.as_posix(), tf.image.encode_jpeg(raw_frame))
            camera_frames[camera].append(frame_path.as_posix())

    return Episode(
        id=episode_id,
        language_instruction_1=language_instructions[0],
        language_instruction_2=language_instructions[1],
        language_instruction_3=language_instructions[2],
        exterior_image_1_left=camera_frames["exterior_image_1_left"],
        exterior_image_2_left=camera_frames["exterior_image_2_left"],
        wrist_image_left=camera_frames["wrist_image_left"],
    )


def append_to_parquet(file_path: Path, data: pd.DataFrame):
    """Append data to a parquet file.

    Args:
        file_path (Path): Path to the parquet file.
        data (pd.DataFrame): Data to append to the parquet file.
    """
    if file_path.exists():
        existing_pf = fastparquet.ParquetFile(file_path)
        fastparquet.write(
            file_path,
            data,
            compression="SNAPPY",
            append=True,
            open_with=existing_pf.open,
        )
    else:
        fastparquet.write(file_path, data, compression="SNAPPY")


def process_droid_dataset(
    dataset_path: Path,
    dataset_name: str,
    output_path: Path,
    split: str,
    chunk_size: int = 20,
):
    """Convert the droid tensorflow dataset to parquet format.

    Args:
        dataset_path (Path): Path to the parent directory of the droid dataset.
        dataset_name (str): Name of the dataset directory.
        output_path (Path): Path to save the processed dataset.
        split (str): Dataset split to convert.
        chunk_size (int, optional): Chunk size to iterate over the tensorflow dataset. Defaults to 20.
    """
    full_dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=dataset_path,
        split="train",
        with_info=True,
    )

    logging.info(f"Dataset info: {dataset_info}")
    num_episodes = len(full_dataset)
    num_chunks = num_episodes // chunk_size
    logging.info(f"Number of episodes: {num_episodes}")
    logging.info(f"Number of dataset chunks: {num_chunks}")
    del full_dataset

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    manifest_path = output_path / MANIFEST_FILE_NAME
    output_files_path = output_path / "images"
    output_files_path.mkdir(parents=True)

    num_episodes_extracted = 0
    with tqdm(total=num_episodes, desc="Processing Episodes") as pbar:
        for i in range(num_chunks):
            chunk_dataset = tfds.load(
                dataset_name,
                data_dir=dataset_path,
                split=f"{split}[{i * chunk_size}:{(i + 1) * chunk_size}]",
            )
            chunk_episodes = []
            for droid_episode in chunk_dataset:
                first_step = tf.data.experimental.get_single_element(
                    droid_episode["steps"].take(1)
                )
                language_instruction = (
                    first_step["language_instruction"].numpy().decode("utf-8")
                )
                if language_instruction:
                    chunk_episodes.append(
                        extract_droid_episode(
                            droid_episode=droid_episode,
                            output_files_path=output_files_path,
                            episode_id=num_episodes_extracted,
                        )
                    )
                    num_episodes_extracted += 1
                pbar.update(1)
            append_to_parquet(
                file_path=manifest_path,
                data=pd.DataFrame([episode.to_pd_row() for episode in chunk_episodes]),
            )


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
        "--split", type=str, default="train", help="Dataset split to convert."
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    dataset_name = args.dataset_name
    output_path = Path(args.output)
    split = args.split
    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Dataset name: {dataset_name}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Split: {split}")
    process_droid_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        output_path=output_path,
        split=split,
    )
