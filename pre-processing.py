import logging
import os
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CAMERAS = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]


def process_episode(
    episode: tf.data.Dataset,
    output_path: Path,
    db_manifest_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
) -> int:
    steps = list(episode["steps"])
    language_instruction = steps[0]["language_instruction"].numpy().decode("utf-8")
    language_instruction_2 = steps[0]["language_instruction_2"].numpy().decode("utf-8")
    language_instruction_3 = steps[0]["language_instruction_3"].numpy().decode("utf-8")
    episode_id = sha256((f"{steps[0]}").encode()).hexdigest()[0:10]
    subsequence_size = len(steps) // num_subsequences

    for i in range(num_subsequences):
        start_idx = i * subsequence_size
        end_idx = start_idx + subsequence_size
        full_subsequence = steps[start_idx:end_idx]
        subsequence = []
        for j in range(
            0, len(full_subsequence), len(full_subsequence) // steps_per_subsequence
        ):
            subsequence.append(full_subsequence[j])

        for camera in CAMERAS:  # iterating over the cameras slows down the process
            images = [step["observation"][camera] for step in subsequence]
            image = tf.concat(images, axis=1)
            file_path = output_path / f"{episode_id}_{camera}_{i}.png"
            tf.io.write_file(file_path.as_posix(), tf.image.encode_png(image))
            with open(db_manifest_path, "a") as f:
                f.write(
                    ",".join(
                        [
                            file_path.absolute().as_posix(),
                            language_instruction,
                            language_instruction_2,
                            language_instruction_3,
                            camera,
                            str(i),
                            str(1 if i == num_subsequences - 1 else 0),
                            "\n",
                        ]
                    )
                )
    return num_subsequences * len(CAMERAS)


def make_new_manifest(reset: bool = True) -> Path:
    db_manifest_path = Path("manifest.csv")
    if reset and db_manifest_path.exists():
        db_manifest_path.unlink()
    db_manifest_path.write_text("path,instruction,camera_name,subsequence,success\n")
    return db_manifest_path


def process_dataset_chunk(
    dataset_chunk: tf.data.Dataset,
    db_manifest_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
    output_path: Path,
) -> tuple[int, int]:
    episodes_processed = 0
    datapoints_created = 0
    for episode in dataset_chunk:
        first_step = tf.data.experimental.get_single_element(episode["steps"].take(1))
        language_instruction = (
            first_step["language_instruction"].numpy().decode("utf-8")
        )
        # at the moment 50k of the 76k success trajectories are annotated https://github.com/droid-dataset/droid/issues/3#issuecomment-2014178692
        logging.info(f"Instruction: {language_instruction}")
        if language_instruction:
            datapoints_created += process_episode(
                episode=episode,
                output_path=output_path,
                db_manifest_path=db_manifest_path,
                num_subsequences=num_subsequences,
                steps_per_subsequence=steps_per_subsequence,
            )
            episodes_processed += 1
    return episodes_processed, datapoints_created


def process_dataset(
    dataset_name: str,
    dataset_dir: str,
    split_size: int,
    db_manifest_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
    output_path: Path,
):
    full_dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=dataset_dir,
        split="train",
        with_info=True,
    )
    logging.info(f"Dataset info: {dataset_info}")
    num_episodes = len(full_dataset)
    num_chunks = num_episodes // split_size
    logging.info(f"Number of episodes: {num_episodes}")
    logging.info(f"Number of datasets chunks to process: {num_chunks}")

    del full_dataset

    total_episode_processed = 0
    total_datapoints_created = 0

    for i in range(num_chunks):
        dataset_chunk = tfds.load(
            dataset_name,
            data_dir=dataset_dir,
            split=f"train[{i * split_size}:{(i + 1) * split_size}]",
        )
        episodes_processed, datapoints_created = process_dataset_chunk(
            dataset_chunk=dataset_chunk,
            db_manifest_path=db_manifest_path,
            num_subsequences=num_subsequences,
            steps_per_subsequence=steps_per_subsequence,
            output_path=output_path,
        )
        total_episode_processed += episodes_processed
        total_datapoints_created += datapoints_created
        logging.info(f"Chunk {i + 1}/{num_chunks} processed")


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--dataset_dir", type=str, default="/media/ismail/WDC")
    argument_parser.add_argument("--split_size", type=int, default=20)
    argument_parser.add_argument("--last_step_shift", type=int, default=5)
    argument_parser.add_argument("--num_subsequences", type=int, default=3)
    argument_parser.add_argument("--steps_per_subsequence", type=int, default=3)
    argument_parser.add_argument("--output_dir", type=str, default="data")
    args = argument_parser.parse_args()

    db_manifest_path = make_new_manifest()
    episodes_processed, datapoints_created = process_dataset(
        dataset_name="droid",
        dataset_dir=args.dataset_dir,
        split_size=args.split_size,
        db_manifest_path=db_manifest_path,
        num_subsequences=args.num_subsequences,
        steps_per_subsequence=args.steps_per_subsequence,
        output_path=Path(args.output_dir),
    )
    logging.info(f"Episodes processed: {episodes_processed}")
    logging.info(f"Datapoints created: {datapoints_created}")
