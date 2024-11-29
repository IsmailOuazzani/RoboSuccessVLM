import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CAMERAS = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]


@dataclass
class InternVLSingleImageData:
    id: int
    image: str
    image_width: int
    image_height: int
    instruction: str
    success: bool

    def to_jsonl(self):
        return {
            "id": self.id,
            "image": self.image,
            "width": self.image_width,
            "height": self.image_height,
            "conversations": [
                {
                    "from": "human",
                    "value": f'<image>\nHas the following task been achieved:"{self.instruction}"? Answer with "yes" or "no" only.',
                },
                {"from": "gpt", "value": "yes" if self.success else "no"},
            ],
        }


def process_episode(
    episode: tf.data.Dataset,
    episode_id: int,
    output_path: Path,
    dataset_file_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
    last_step_shift: int,
) -> int:
    """Processes an episode by creating subsequences and saving images to disk.

    Args:
        episode (tf.data.Dataset): Episode to process.
        episode_id (int): Episode ID.
        output_path (Path): Path to save the images.
        dataset_file_path (Path): Path to the dataset manifest file.
        num_subsequences (int): Number of subsequences to create per episode.
        steps_per_subsequence (int): Number of steps per subsequence to sample from.
        last_step_shift (int): Shift for the last step in the subsequence (recording stops after the trajectory finishes).

    Returns:
        int: Number of datapoints created.
    """

    steps = list(episode["steps"])
    language_instructions = [
        steps[0]["language_instruction"].numpy().decode("utf-8"),
        steps[0]["language_instruction_2"].numpy().decode("utf-8"),
        steps[0]["language_instruction_3"].numpy().decode("utf-8"),
    ]
    subsequence_size = len(steps) // num_subsequences

    data_points: list[InternVLSingleImageData] = []

    for i in range(num_subsequences):
        full_subsequence = steps[i * subsequence_size : (i + 1) * subsequence_size]
        indices = np.linspace(
            0,
            len(full_subsequence) - 1 - last_step_shift,
            steps_per_subsequence,
            dtype=int,
        ).tolist()
        subsequence = [full_subsequence[j] for j in indices]

        for camera in CAMERAS:  # iterating over the cameras slows down the process
            images = [step["observation"][camera] for step in subsequence]
            image = tf.concat(images, axis=1)
            file_path = output_path / "images" / f"{episode_id}_{camera}_{i}.png"
            tf.io.write_file(file_path.as_posix(), tf.image.encode_png(image))

            for language_instruction in language_instructions:
                if language_instruction != "":
                    data_points.append(
                        InternVLSingleImageData(
                            id=episode_id + len(data_points),
                            image=file_path.relative_to(output_path).as_posix(),
                            image_width=image.shape[1],
                            image_height=image.shape[0],
                            instruction=language_instruction,
                            success=True if i == num_subsequences - 1 else False,
                        )
                    )

    with open(dataset_file_path, "a") as f:
        for data_point in data_points:
            f.write(json.dumps(data_point.to_jsonl()) + "\n")

    return len(data_points)


def process_dataset(
    dataset_name: str,
    dataset_dir: str,
    chunk_size: int,
    dataset_file_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
    last_step_shift: int,
    output_path: Path,
    max_episodes: int,
) -> tuple[int, int]:
    """
    Processes a dataset by splitting it into chunks and processing each episode within each chunk.

    Arguments:
        dataset_name (str): The name of the directory containing the dataset.
        dataset_dir (str): The parent directory containing the dataset directory.
        chunk_size (int): Number of episodes per dataset chunk to process.
        dataset_file_path (Path): The path to the dataset manifest file.
        num_subsequences (int): Number of subsequences to create per episode.
        steps_per_subsequence (int): Number of steps per subsequence to sample from.
        last_step_shift (int): Shift for the last step in the subsequence (recording stops after the trajectory finishes).
        output_path (Path): Path where output data will be stored.
        max_episodes (int): Maximum number of episodes to process.
    Returns:
        tuple[int, int]: Total episodes processed and total datapoints created.
    """
    full_dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=dataset_dir,
        split="train",
        with_info=True,
    )
    logging.info(f"Dataset info: {dataset_info}")
    num_episodes = len(full_dataset)
    num_chunks = num_episodes // chunk_size
    logging.info(f"Number of episodes: {num_episodes}")
    logging.info(f"Number of dataset chunks to process: {num_chunks}")

    del full_dataset

    num_episodes_processed = 0
    num_datapoints_created = 0

    for i in range(num_chunks):
        dataset_chunk = tfds.load(
            dataset_name,
            data_dir=dataset_dir,
            split=f"train[{i * chunk_size}:{(i + 1) * chunk_size}]",
        )

        for episode in dataset_chunk:
            first_step = tf.data.experimental.get_single_element(
                episode["steps"].take(1)
            )
            language_instruction = (
                first_step["language_instruction"].numpy().decode("utf-8")
            )
            logging.info(f"Instruction: {language_instruction}")
            if language_instruction:
                num_datapoints_created += process_episode(
                    episode_id=num_datapoints_created,
                    episode=episode,
                    output_path=output_path,
                    dataset_file_path=dataset_file_path,
                    num_subsequences=num_subsequences,
                    steps_per_subsequence=steps_per_subsequence,
                    last_step_shift=last_step_shift,
                )
                num_episodes_processed += 1
                if num_episodes_processed >= max_episodes:
                    return num_episodes_processed, num_datapoints_created
        logging.info(f"Chunk {i + 1}/{num_chunks} processed")

    return num_episodes_processed, num_datapoints_created


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/media/ismail/WDC",
        help="Path to the parent directory of the dataset.",
    )
    argument_parser.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="Number of episodes per dataset chunk to process.",
    )
    argument_parser.add_argument(
        "--last_step_shift",
        type=int,
        default=5,
        help="Shift for the last step in the subsequence (since the video usually goes on after the robot finished).",
    )
    argument_parser.add_argument(
        "--num_subsequences",
        type=int,
        default=3,
        help="Number of subsequences to create per episode.",
    )
    argument_parser.add_argument(
        "--steps_per_subsequence",
        type=int,
        default=3,
        help="Number of frames per subsequence to sample.",
    )
    argument_parser.add_argument(
        "--output_dir", type=str, default="data", help="Path to the output directory."
    )
    argument_parser.add_argument(
        "--max_episodes",
        type=int,
        default=1000,
        help="Maximum number of episodes to process.",
    )
    args = argument_parser.parse_args()

    dataset_file_path = Path(f"{args.output_dir}/dataset.jsonl")
    dataset_file_path.unlink(missing_ok=True)
    dataset_file_path.touch()

    episodes_processed, datapoints_created = process_dataset(
        dataset_name="droid",
        dataset_dir=args.dataset_dir,
        chunk_size=args.chunk_size,
        dataset_file_path=dataset_file_path,
        num_subsequences=args.num_subsequences,
        steps_per_subsequence=args.steps_per_subsequence,
        last_step_shift=args.last_step_shift,
        output_path=Path(args.output_dir),
        max_episodes=args.max_episodes,
    )
    logging.info(f"Episodes processed: {episodes_processed}")
    logging.info(f"Datapoints created: {datapoints_created}")

    dataset_name = f"droid_{args.num_subsequences}_{args.steps_per_subsequence}_{args.last_step_shift}_{datapoints_created}"
    meta_file_path = f"{dataset_name}.json"
    with open(meta_file_path, "w") as f:
        f.write(
            json.dumps(
                {
                    dataset_name: {
                        "root": args.output_dir,
                        "annotation": dataset_file_path.relative_to(
                            args.output_dir
                        ).as_posix(),
                        "repeat_time": 1,
                        "length": datapoints_created,
                    }
                }
            )
        )

    config_file = Path(args.output_dir) / "README"
    with open(config_file, "w") as f:
        f.write(json.dumps(vars(args), indent=4))

    logging.info(f"Dataset created: {dataset_name}")
