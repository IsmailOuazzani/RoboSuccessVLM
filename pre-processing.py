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


def process_episode(
    episode: tf.data.Dataset,
    output_path: Path,
    language_instruction: str,
    db_manifest_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
) -> int:
    steps = list(episode["steps"])
    instruction_hash = sha256((f"{language_instruction}").encode()).hexdigest()[0:10]
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

        images = [step["observation"]["wrist_image_left"] for step in subsequence]
        image = tf.concat(images, axis=1)
        file_path = output_path / f"{instruction_hash}_{i}.png"
        tf.io.write_file(file_path.as_posix(), tf.image.encode_png(image))
        with open(db_manifest_path, "a") as f:
            f.write(f'{file_path.absolute().as_posix()},"{language_instruction}",{i}\n')
    return num_subsequences


def make_new_manifest(reset: bool = True) -> Path:
    db_manifest_path = Path("manifest.csv")
    if reset and db_manifest_path.exists():
        db_manifest_path.unlink()
    db_manifest_path.write_text("path,instruction,subsequence\n")
    return db_manifest_path


def process_dataset(
    dataset: tf.data.Dataset,
    db_manifest_path: Path,
    num_subsequences: int,
    steps_per_subsequence: int,
    output_path: Path,
) -> tuple[int, int]:
    episodes_processed = 0
    datapoints_created = 0
    for episode in dataset:  # TODO: move to a process_dataset function
        first_step = tf.data.experimental.get_single_element(episode["steps"].take(1))
        language_instruction = (
            first_step["language_instruction"].numpy().decode("utf-8")
        )
        # at the moment 50k of the 76k success trajectories are annotated https://github.com/droid-dataset/droid/issues/3#issuecomment-2014178692
        logging.info(f"Instruction: {language_instruction}")
        if language_instruction:
            # TODO: check for success
            datapoints_created += process_episode(
                episode=episode,
                output_path=output_path,
                language_instruction=language_instruction,
                db_manifest_path=db_manifest_path,
                num_subsequences=num_subsequences,
                steps_per_subsequence=steps_per_subsequence,
            )
            episodes_processed += 1
    return episodes_processed, datapoints_created


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--dataset_dir", type=str, default="/media/ismail/WDC")
    argument_parser.add_argument("--split_size", type=int, default=500)
    argument_parser.add_argument("--last_step_shift", type=int, default=5)
    argument_parser.add_argument("--num_subsequences", type=int, default=3)
    argument_parser.add_argument("--steps_per_subsequence", type=int, default=4)
    argument_parser.add_argument("--output_dir", type=str, default="data")
    args = argument_parser.parse_args()

    dataset, dataset_info = tfds.load(
        "droid",
        data_dir=args.dataset_dir,
        split=f"train[:{args.split_size}]",
        with_info=True,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logging.info(dataset_info)

    num_episodes = len(dataset)
    episodes = dataset.take(num_episodes)
    logging.info(f"Number of episodes in the dataset: {num_episodes}")

    db_manifest_path = make_new_manifest()
    episodes_processed, datapoints_created = process_dataset(
        dataset=episodes,
        db_manifest_path=db_manifest_path,
        num_subsequences=args.num_subsequences,
        steps_per_subsequence=args.steps_per_subsequence,
        output_path=Path(args.output_dir),
    )
    logging.info(f"Episodes processed: {episodes_processed}")
    logging.info(f"Datapoints created: {datapoints_created}")
