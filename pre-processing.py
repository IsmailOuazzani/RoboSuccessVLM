import json
import logging
import os
import zipfile
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CAMERAS = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]


@dataclass
class InternVLDataPoint:
    id: int
    image: list[str]
    width_list: list[int]
    height_list: list[int]
    instruction: str
    success_list: list[bool]
    combined: bool = False

    def to_jsonl(self):
        conversations = []

        if self.combined:
            conversations.extend(
                [
                    {
                        "from": "human",
                        "value": f'{" ".join(["Step "+ str(i)+": <image>" for i in range(len(self.image))])}\nHas the following task been achieved:"{self.instruction}"? Answer with "yes" or "no" only.',
                    },
                    {"from": "gpt", "value": "yes" if all(self.success_list) else "no"},
                ]
            )
        else:
            for i in range(len(self.image)):
                conversations.extend(
                    [
                        {
                            "from": "human",
                            "value": f'<image>\nHas the following task been achieved:"{self.instruction}"? Answer with "yes" or "no" only.',
                        },
                        {
                            "from": "gpt",
                            "value": "yes" if self.success_list[i] else "no",
                        },
                    ]
                )
        return {
            "id": self.id,
            "image": self.image,
            "width_list": self.width_list,
            "height_list": self.height_list,
            "conversations": conversations,
        }


def process_episode(
    episode: tf.data.Dataset,
    episode_id: int,
    output_path: Path,
    dataset_file_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
    last_step_shift: int,
    conversation_format: str,
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
        conversation_format (str): The format of the conversation.

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

    data_points: list[InternVLDataPoint] = []

    if conversation_format == "single_turn":
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
                            InternVLDataPoint(
                                id=episode_id + len(data_points),
                                image=[file_path.relative_to(output_path).as_posix()],
                                width_list=[image.shape[1]],
                                height_list=[image.shape[0]],
                                instruction=language_instruction,
                                success_list=[
                                    True if i == num_subsequences - 1 else False
                                ],
                            )
                        )
    else:
        for language_instruction in language_instructions:
            if language_instruction != "":
                for camera in CAMERAS:
                    subsequence_processed = []
                    for i in range(num_subsequences):
                        full_subsequence = steps[
                            i * subsequence_size : (i + 1) * subsequence_size
                        ]
                        indices = np.linspace(
                            0,
                            len(full_subsequence) - 1 - last_step_shift,
                            steps_per_subsequence,
                            dtype=int,
                        ).tolist()
                        subsequence = [full_subsequence[j] for j in indices]

                        images = [step["observation"][camera] for step in subsequence]
                        image = tf.concat(images, axis=1)
                        file_path = (
                            output_path / "images" / f"{episode_id}_{camera}_{i}.png"
                        )
                        tf.io.write_file(
                            file_path.as_posix(), tf.image.encode_png(image)
                        )

                        subsequence_processed.append(
                            {
                                "image": file_path.relative_to(output_path).as_posix(),
                                "width": image.shape[1],
                                "height": image.shape[0],
                            }
                        )
                if conversation_format == "single_turn_combined":
                    data_points.append(
                        InternVLDataPoint(
                            id=episode_id + len(data_points),
                            image=[
                                subsequence["image"]
                                for subsequence in subsequence_processed
                            ],
                            width_list=[
                                subsequence["width"]
                                for subsequence in subsequence_processed
                            ],
                            height_list=[
                                subsequence["height"]
                                for subsequence in subsequence_processed
                            ],
                            instruction=language_instruction,
                            success_list=[True],
                            combined=True,
                        )
                    )
                elif conversation_format == "multi_turn":
                    data_points.append(
                        InternVLDataPoint(
                            id=episode_id + len(data_points),
                            image=[
                                subsequence["image"]
                                for subsequence in subsequence_processed
                            ],
                            width_list=[
                                subsequence["width"]
                                for subsequence in subsequence_processed
                            ],
                            height_list=[
                                subsequence["height"]
                                for subsequence in subsequence_processed
                            ],
                            instruction=language_instruction,
                            success_list=[
                                True if i == num_subsequences - 1 else False
                                for i in range(num_subsequences)
                            ],
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
    conversation_format: str,
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
        conversation_format (str): The format of the conversation.
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
    logging.info(f"Number of dataset chunks: {num_chunks}")

    del full_dataset

    num_episodes_processed = 0
    num_datapoints_created = 0

    with tqdm(total=max_episodes, desc="Processing Episodes") as pbar:
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
                logging.debug(f"Instruction: {language_instruction}")
                if language_instruction:
                    num_datapoints_created += process_episode(
                        episode_id=num_datapoints_created,
                        episode=episode,
                        output_path=output_path,
                        dataset_file_path=dataset_file_path,
                        num_subsequences=num_subsequences,
                        steps_per_subsequence=steps_per_subsequence,
                        last_step_shift=last_step_shift,
                        conversation_format=conversation_format,
                    )
                    num_episodes_processed += 1
                    pbar.update(1)
                    if num_episodes_processed >= max_episodes:
                        return num_episodes_processed, num_datapoints_created
            logging.debug(f"Chunk {i + 1}/{num_chunks} processed")

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
        default=1,
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
        help="Maximum number of episodes to process. Episodes without instructions are skipped and do not count.",
    )
    argument_parser.add_argument(
        "--conversation_format",
        type=str,
        default="single_turn",
        choices={"single_turn", "single_turn_combined", "multi_turn"},
        help="Format of the conversation to use.",
    )
    argument_parser.add_argument(
        "--export_dataset",
        action="store_true",
        help="Export the dataset to a zip file.",
    )

    args = argument_parser.parse_args()

    dataset_file_path = Path(f"{args.output_dir}/dataset.jsonl")
    dataset_file_path.unlink(missing_ok=True)
    dataset_file_path.touch()

    logging.getLogger("absl").setLevel(logging.WARNING)

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
        conversation_format=args.conversation_format,
    )
    logging.info(f"Episodes processed: {episodes_processed}")
    logging.info(f"Datapoints created: {datapoints_created}")

    dataset_name = f"droid_{args.num_subsequences}_{args.steps_per_subsequence}_{args.last_step_shift}_{args.conversation_format}_{datapoints_created}"
    meta_file_path = f"{dataset_name}.json"
    with open(meta_file_path, "w") as f:
        f.write(
            json.dumps(
                {
                    dataset_name: {
                        "root": "shell/data/" + args.output_dir,
                        "annotation": "shell/data/"
                        + dataset_file_path.relative_to(args.output_dir).as_posix(),
                        "repeat_time": 1,
                        "length": datapoints_created,
                        "data_augment": False,
                    }
                }
            )
        )

    config_file = Path(args.output_dir) / "README"
    with open(config_file, "w") as f:
        f.write(json.dumps(vars(args), indent=4))

    logging.info(f"Dataset created: {dataset_name}")

    if args.export_dataset:
        with zipfile.ZipFile(f"{dataset_name}.zip", "w") as zipf:
            zipf.write(dataset_file_path)
            zipf.write(meta_file_path)
            zipf.write(config_file)
            for image_file in Path(args.output_dir).rglob("*.png"):
                zipf.write(image_file)

        logging.info(f"Dataset exported: {dataset_name}.zip")
