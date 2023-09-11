# continual_learning_experiment.py
import os, json, time
from datetime import datetime
from pathlib import Path
import subprocess
import dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from annotations import db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import sys

sys.path.append("/Users/simon/Documents/Projects/TFM/")
sys.path.append("/Users/simon/Documents/Projects/TFM/bacili_detection/detr")

from bacili_detection.utils.evaluate import evaluate_trained_model
from bacili_detection.detr.datasets.tb_bacillus import TBBacilliDataset


def continual_learning_experiment(
    dataset_file: str,
    frac: list,
    image_dir: str,
    output_dir: str,
    checkpoint: str,
    log_file: str,
    epochs: int,
    num_classes: int,
    batch_size: int,
    device: str,
    tags: str = "incremental_training",
    start_with: str = None,
):
    dotenv.load_dotenv(".env")
    session = db.get_session(os.environ.get("DATABASE_URI"))

    holdout_artifacts = (
        session.query(db.Artifact)
        .join(db.Project)
        .join(db.ArtifactTag, isouter=True)
        .where(db.Project.name == "Bacilli Detection")
        .group_by(db.Artifact.id)
        .where(db.ArtifactTag.tag == "holdout")
        .all()
    )

    total_num_images = len(holdout_artifacts)
    artifacts_for_increment = []

    if total_num_images == 0:
        print("No images to train on, exiting...")
        return

    print(
        f"Starting continual learning experiment for dataset: {dataset_file} at: {datetime.now():%Y-%m-%d %H:%M:%S}"
    )
    print("Total number of images: ", total_num_images)
    print("Number of classes: ", num_classes)
    print("Number of epochs: ", epochs)
    print("Batch size: ", batch_size)
    print("Device: ", device)
    print("Tags: ", tags)
    print("Fractions of images to train on: ", frac)
    print("Output directory: ", output_dir)
    print("Checkpoint: ", checkpoint)
    print("Log file: ", log_file)
    print("Image directory: ", image_dir)
    print("-----------------------------------------")
    for i, percentage in tqdm(
        enumerate(frac), desc="Training on incremental fractions of images"
    ):
        if i == 0 and start_with:
            # Start with a pretrained model
            checkpoint = start_with

        num_images = int(percentage * total_num_images)

        # Select new artifacts from holdout for this training iteration
        new_artifacts_for_increment = np.random.choice(
            holdout_artifacts, size=num_images, replace=False
        )
        artifacts_for_increment += list(new_artifacts_for_increment)

        # Remove the selected artifacts from holdout
        holdout_artifacts = list(
            set(holdout_artifacts) - set(new_artifacts_for_increment)
        )

        # Update tags in the database for selected artifacts
        for artifact in artifacts_for_increment:
            tag = db.ArtifactTag(tag="incremental_training", artifact_id=artifact.id)
            session.add(tag)

        session.commit()

        # Call bash script to train
        output_dir = Path(f"{output_dir}/experiment_{i}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # save the artifacts ids in a file
        artifacts_ids = [artifact.id for artifact in artifacts_for_increment]
        with open(f"{output_dir}/artifacts_ids.json", "w") as f:
            json.dump(
                {
                    "iteration_number": i,
                    "num_images": num_images,
                    "percentage": percentage,
                    "artifacts_ids": artifacts_ids,
                    "timestamp": f"{datetime.now()}",
                    "tags_to_train_on": tags,
                    "number_of_objects_in_db_with_tag": len(TBBacilliDataset(tags))
                },
                f,
                indent=4,
            )

        # the train script should be in the same directory as this script
        train_sh_dir = os.path.dirname(os.path.realpath(__file__))
        proc = subprocess.Popen(
            [
                f"{train_sh_dir}/train.sh",
                dataset_file,
                image_dir,
                str(output_dir),
                checkpoint,
                log_file,
                epochs,
                num_classes,
                batch_size,
                device,
                tags,
            ],
            stdout=subprocess.PIPE,
        )
        while True:
            output = proc.stdout.readline()
            if output == "" and proc.poll() is not None:
                break
            if output:
                print(output.strip())
            rc = proc.poll()

        # Evaluate and store results
        evaluate_trained_model(output_dir, device)

    print("Finished continual learning experiment successfully!!!")


if __name__ == "__main__":
    dataset_file = "bacilli_detection"
    image_dir = "/Users/simon/Documents/Projects/TFM/"
    output_dir = "bacili_detection/detr/outputs"
    checkpoint = "checkpoint.pth"
    log_file = "train.log"
    epochs = "100"
    num_classes = "2"
    batch_size = "2"
    device = "mps"
    tags = "incremental_training"
    frac = [0, 0.2, 0.5, 0.75, 1]
    start_with = "/Users/simon/Documents/Projects/TFM/bacili_detection/detr/detr-r50_no-class-head.pth"

    print("Starting continual learning experiment for dataset: ", dataset_file, "at workdir: ", os.getcwd())
    continual_learning_experiment(
        dataset_file=dataset_file,
        frac=frac,
        image_dir=image_dir,
        output_dir=output_dir,
        checkpoint=checkpoint,
        log_file=log_file,
        epochs=epochs,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
        tags=tags,
        start_with=start_with,
    )
