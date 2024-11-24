import logging
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RAW_DATASET_DIR = "/media/ismail/WDC"
PROCESSED_DATASET_DIR = "/media/ismail/BIG/datasets/droid-processed"

SPLIT_SIZE = 500
LAST_STEP_SHIFT = 5


def process_episode(episode: tf.data.Dataset, output_path: Path):
    first_step = tf.data.experimental.get_single_element(episode["steps"].take(1))
    middle_step = tf.data.experimental.get_single_element(
        episode["steps"].skip(len(episode["steps"]) // 2).take(1)
    )
    last_step = tf.data.experimental.get_single_element(
        episode["steps"].skip(len(episode["steps"]) - LAST_STEP_SHIFT).take(1)
    )
    first_image = first_step["observation"]["wrist_image_left"]
    middle_image = middle_step["observation"]["wrist_image_left"]
    last_image = last_step["observation"]["wrist_image_left"]
    image = tf.concat([first_image, middle_image, last_image], axis=1)
    image = tf.image.encode_png(image)
    image = tf.io.write_file(output_path.as_posix(), image)


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
            process_episode(episode=episode, output_path=Path("test.png"))
