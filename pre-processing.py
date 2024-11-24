import logging
import os
from hashlib import sha256
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RAW_DATASET_DIR = "/media/ismail/WDC"
PROCESSED_DATASET_DIR = "/media/ismail/BIG/datasets/droid-processed"

SPLIT_SIZE = 20
LAST_STEP_SHIFT = 5


def process_episode(
    episode: tf.data.Dataset,
    output_path: Path,
    language_instruction: str,
    num_subsequences: int = 4,
    steps_per_subsequence: int = 3,
):
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


if __name__ == "__main__":
    dataset, dataset_info = tfds.load(
        "droid",
        data_dir="/media/ismail/WDC",
        split=f"train[:{SPLIT_SIZE}]",
        with_info=True,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logging.info(dataset_info)

    num_episodes = len(dataset)
    episodes = dataset.take(num_episodes)
    logging.info(f"Number of episodes in the dataset: {num_episodes}")

    for episode in dataset:
        first_step = tf.data.experimental.get_single_element(episode["steps"].take(1))
        language_instruction = (
            first_step["language_instruction"].numpy().decode("utf-8")
        )
        # at the moment 50k of the 76k success trajectories are annotated https://github.com/droid-dataset/droid/issues/3#issuecomment-2014178692
        logging.info(f"Instruction: {language_instruction}")
        if language_instruction:
            # TODO: check for success
            process_episode(
                episode=episode,
                output_path=Path("./temp/"),
                language_instruction=language_instruction,
            )
