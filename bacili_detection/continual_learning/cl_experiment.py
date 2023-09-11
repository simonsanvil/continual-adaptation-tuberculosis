# continual_learning_experiment.py
import os, json, signal, atexit
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

sys.path.append("/content/continual-adaptation-tuberculosis/bacili_detection/detr")
sys.path.append("/content/continual-adaptation-tuberculosis")

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
    incremental_epochs: bool = True,
    from_scratch: bool = False,
):
    dotenv.load_dotenv(".env")
    session = db.get_session(os.environ.get("DATABASE_URI"))
    if from_scratch:
        init_experiment_db(session, image_dir)
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
    epochs_per_increment = (
        epochs if incremental_epochs is True else int(incremental_epochs)
    )

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
    prev_checkpoint = None
    for i, percentage in tqdm(
        enumerate(frac), desc="Training on incremental fractions of images"
    ):
        if percentage > 0:
            num_images = percentage
        else:
            num_images = int(percentage * total_num_images)

        # Select new artifacts from holdout for this training iteration
        new_artifacts_for_increment = np.random.choice(
            holdout_artifacts,
            size=min(num_images, len(holdout_artifacts)),
            replace=False,
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

        # If this is the first iteration, start with a pretrained model
        if prev_checkpoint is None and start_with:
            prev_checkpoint = start_with

        # Call bash script to train
        output_dir_exp = Path(f"{output_dir}/experiment_{i}").resolve()
        output_dir_exp.mkdir(parents=True, exist_ok=True)
        # save the artifacts ids in a file
        artifacts_ids = [artifact.id for artifact in artifacts_for_increment]
        with open(f"{output_dir_exp}/experiment_details.json", "w") as f:
            json.dump(
                {
                    "iteration_number": i,
                    "num_images_started": total_num_images,
                    "num_images_added": num_images,
                    "percentage": percentage,
                    "new_artifacts_ids": artifacts_ids,
                    "timestamp_start": f"{datetime.now()}",
                    "tags_to_train_on": tags,
                    "number_of_objects_in_train": len(
                        TBBacilliDataset(tags, db_session=session, image_dir=image_dir)
                    ),
                    "checkpoint_trained_on": prev_checkpoint,
                },
                f,
                indent=4,
            )

        # the train script should be in the same directory as this script
        train_sh_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [
            f"{train_sh_dir}/train.sh",
            dataset_file,
            image_dir,
            str(output_dir_exp),
            prev_checkpoint,
            log_file,
            epochs,
            num_classes,
            batch_size,
            device,
            tags,
        ]
        print(f"started running training iteration {i} at: {datetime.now():%Y-%m-%d %H:%M:%S}")
        tstart = datetime.now()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # make sure the process is killed when the python script is killed
        atexit.register(proc.kill)
        while True:
            output = proc.stdout.readline()
            if proc.poll() is not None:
                break
            if output:
                print(output.strip())
            rc = proc.poll()  # returns None while subprocess is running
        print(f"Finished training iteration {i} at: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Training iteration {i} took: {datetime.now() - tstart}")
        # reload the experimental_details.json file and add the timestamp_end
        with open(f"{output_dir_exp}/experiment_details.json", "r") as f:
            experiment_details = json.load(f)
        experiment_details["timestamp_end"] = f"{datetime.now()}"
        with open(f"{output_dir_exp}/experiment_details.json", "w") as f:
            json.dump(experiment_details, f, indent=4)
        # Evaluate and store results
        evaluate_trained_model(
            checkpoint_dir=Path(prev_checkpoint).parent if i != 0 else output_dir_exp,
            file=f"{output_dir_exp}/eval.csv",
            device=device,
            image_dir=image_dir,
        )
        prev_checkpoint = f"{output_dir_exp}/{checkpoint}"

        if incremental_epochs:
            epochs += epochs_per_increment

    print("Finished continual learning experiment successfully!!!")


def init_experiment_db(session, image_dir):
    """
    Initialize the holdout and incremental_training tags in the database
    """
    # first make sure that no holdout or incremental_training tags exist
    holdout_ds = TBBacilliDataset("holdout", db_session=session, image_dir=image_dir)
    print("Found {} holdout artifacts to begin with".format(len(holdout_ds)))
    train_cl_ds = TBBacilliDataset(
        "incremental_training", db_session=session, image_dir=image_dir
    )
    print(
        "Found {} incremental_training artifacts to begin with".format(len(train_cl_ds))
    )
    tags_deleted = 0
    for imod in (holdout_ds + train_cl_ds)._images:
        artifact = imod.artifact
        for tag in artifact.tags:
            if tag.tag == "holdout":
                session.delete(tag)
                tags_deleted += 1
            if tag.tag == "incremental_training":
                session.delete(tag)
                tags_deleted += 1
    session.commit()
    print("Deleted {} holdout and incremental_training tags".format(tags_deleted))
    # now add the holdout tag to half amount of training images
    tr_ds = TBBacilliDataset("train", db_session=session, image_dir=image_dir)
    train_artifacts = [imod.artifact for imod in tr_ds._images]
    inds = np.arange(len(train_artifacts))
    holdout_artifacts_inds = np.random.choice(
        inds, size=len(train_artifacts) // 2, replace=False
    )
    # tag them as holdout
    for i in holdout_artifacts_inds:
        artifact = train_artifacts[i]
        newtag = db.ArtifactTag(tag="holdout", artifact_id=artifact.id)
        session.add(newtag)
    session.commit()
    # add the tag 'incremental_training' tag to the rest
    for i in inds:
        if i not in holdout_artifacts_inds:
            artifact = train_artifacts[i]
            newtag = db.ArtifactTag(tag="incremental_training", artifact_id=artifact.id)
            session.add(newtag)
    session.commit()


if __name__ == "__main__":
    # dataset_file = "bacilli_detection"
    # image_dir = "/Users/simon/Documents/Projects/TFM/"
    # output_dir = "bacili_detection/detr/outputs"
    # checkpoint = "checkpoint.pth"
    # log_file = "train.log"
    # epochs = "1"
    # num_classes = "2"
    # batch_size = "2"
    # device = "mps"
    # tags = "incremental_training"
    # # frac = [0, 0.2, 0.5, 0.75, 1]
    # frac = [0, 0.2, 0.3, 0.25, 0.25]
    # start_with = "/Users/simon/Documents/Projects/TFM/bacili_detection/detr/detr-r50_no-class-head.pth"
    dataset_file = "bacilli_detection"
    image_dir = "/gdrive/MyDrive/Projects/TFM/"
    checkpoint = "checkpoint.pth"
    log_file = "train.log"
    epochs = "25"
    num_classes = "2"
    batch_size = "2"
    tags = "incremental_training"
    frac = [0.1, 0.2, 0.2, 0.25, 0.25]
    start_with = "detr-r50_no-class-head.pth"
    device = "cuda"
    output_dir = "/gdrive/MyDrive/Projects/TFM/outputs/"
    DATA_PATH = Path("/gdrive/MyDrive/Projects/TFM/data")
    DB_PATH = DATA_PATH / "annotations.db"
    from_scratch = False

    os.environ["DATABASE_URI"] = f"sqlite:///{DB_PATH}"
    session = db.get_session(os.environ["DATABASE_URI"])
    print(
        "Starting continual learning experiment for dataset: ",
        dataset_file,
        "at workdir: ",
        os.getcwd(),
    )
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
        incremental_epochs=True,
        from_scratch=from_scratch,
    )
