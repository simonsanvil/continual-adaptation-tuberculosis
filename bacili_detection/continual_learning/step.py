"""
a continual learning step:

step('train', **kwargs) -> trains a model on the set with tag 'train'
step(0, **kwargs) -> trains a model on the first holdout
step(1, **kwargs) -> trains a model on the second holdout
step([0, 1], **kwargs) -> trains a model on the first and second holdouts
step(-1, **kwargs) -> trains a model on the last holdout
...
"""

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
import torch
from tqdm import tqdm
import sys

sys.path.append("/content/continual-adaptation-tuberculosis/bacili_detection/detr")
sys.path.append("/content/continual-adaptation-tuberculosis")

from bacili_detection.utils.evaluate import evaluate_trained_model
from bacili_detection.detr.datasets.tb_bacillus import TBBacilliDataset
from bacili_detection.continual_learning.holdouts import query_holdouts_at_step
from ml_collections import ConfigDict

# image_dir=$2
# output_dir=$3
# resume=$4
# logging_file=$5
# epochs=$6
# batch_size=$8
# device=$9
# tags=${10}


incremental_config = ConfigDict({
    'lr' : 1e-4, #0.0001,
    'weight_decay' : 1e-4,
    'lr_backbone' : 1e-5,
    'epochs' :5,
})
nonincremental_config = ConfigDict({
    'lr' : 1e-4, #0.0001,
    'weight_decay' : 1e-4,
    'lr_backbone' : 1e-5,
    'epochs' :8,
})

def step(
    holdout_step:int,
    experiment_dir:str,
    model_checkpoint:str='checkpoint.pth',
    incremental:bool=True,
    holdout_tag:str="holdout",
    session=None,
    device:str="cuda",
    batch_size:int=2,
    image_dir:str="",
    **kwargs
):
    dotenv.load_dotenv(".env")
    if session is None:
        session = db.get_session(os.environ.get("DATABASE_URI"))
    # make tags to train on
    tags_to_train_on = []
    if not incremental or holdout_step==-1:
        # not incremental means retraining on all data, including holdouts
        config = nonincremental_config.copy_and_resolve_references()
        tags_to_train_on += [f"{holdout_tag}:{-1}"]
        for i in range(holdout_step + 1):
            tags_to_train_on += [f"{holdout_tag}:{i}"]
    else:
        # incremental means retraining only on the holdout step
        config = incremental_config.copy_and_resolve_references()
        tags_to_train_on += [f"{holdout_tag}:{holdout_step}"]
    
    # config.epochs 
    # make the holdout step directory
    tagsteps = '_'.join(str(i) for i in range(-1,holdout_step + 1)) if not incremental else str(holdout_step)
    dirpath = Path(experiment_dir) / f"steps_{tagsteps}"
    while dirpath.exists():
        # we need to make sure we don't overwrite a previous step
        dirpath = Path(str(dirpath) + '_new')

    dirpath.mkdir(exist_ok=True, parents=True)
    dirpath = dirpath.resolve()
    if model_checkpoint=='checkpoint.pth' and holdout_step >= 0 and incremental:
        # if we are not starting from scratch, we need to find the previous checkpoint
        model_checkpoint = Path(experiment_dir) / f"steps_{holdout_step-1}" / model_checkpoint
    elif model_checkpoint=='checkpoint.pth' and incremental:
        model_checkpoint = dirpath / model_checkpoint
    else:
        model_checkpoint = Path(model_checkpoint)
    
    checkpoint_epochs = get_epoch_from_checkpoint(model_checkpoint)
    print("checkpoint_epochs", checkpoint_epochs)
    # set the number of epochs to train for based on the checkpoint
    config['epochs'] = checkpoint_epochs + config.epochs
    # make the config file
    config['tags'] = ','.join(tags_to_train_on)
    config['output_dir'] = str(dirpath)
    config['device'] = device
    config['batch_size'] = batch_size
    config['image_dir'] = str(Path(image_dir).resolve())
    config['resume'] = str(model_checkpoint.resolve())
    config['logging_file'] = str(dirpath / 'log.txt')
    config['holdout_step'] = holdout_step
    config['incremental'] = incremental
    config.update(kwargs)
    save_config(config, dirpath, model_checkpoint, session=session)
    # run the training script
    print("attempting to train epochs", config.epochs)
    config = run_train_script(config)
    # save the config again
    save_config(config, dirpath, model_checkpoint, session=session)
    # evaluate the model
    evaluate_trained_model(
        str(config.output_dir),
        file=str(dirpath / 'eval.csv'),
        device=config.device,
        image_dir=config.image_dir,
    )
    print("Results saved to: ", dirpath)


def get_epoch_from_checkpoint(checkpoint:str):
    checkpoint = torch.load(checkpoint, map_location='cuda')
    return checkpoint.get('epoch', 0)


def save_config(config, dirpath, checkpoint:str, session=None):

    ds = TBBacilliDataset(config.tags.split(','), db_session=session, image_dir=config.image_dir)
    with open(f"{dirpath}/experiment_details.json", "w") as f:
        json.dump(
            {
                "holdout_step": config.holdout_step,
                "incremental": config.incremental,
                "num_train_instances": len(ds),
                "new_artifacts_ids": [imod.artifact.id for imod in ds._images],
                "timestamp_start": f"{datetime.now()}",
                "tags_to_train_on": config.tags.split(','),
                "checkpoint_trained_on": checkpoint,
                "config": config,
            },
            f,
            indent=4,
            default=str
        )

def run_train_script(config):
    train_sh_dir = os.path.dirname(os.path.realpath(__file__))
    cmd = [
        f"{train_sh_dir}/train.sh",
        config.image_dir,
        config.output_dir,
        config.resume,
        config.epochs,
        config.batch_size,
        config.device,
        config.tags,
    ]
    print(f"Running command: {' '.join([str(s) for s in cmd])}")
    print(f"started running training iteration {config.holdout_step} at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    tstart = datetime.now()
    cmd = [str(s) for s in cmd]
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
    print(f"Finished training iteration {config.holdout_step} at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Training step {config.holdout_step} took: {datetime.now() - tstart}")
    config.resume = f"{config.output_dir}/checkpoint.pth"
    config['timestamp_end'] = f"{datetime.now()}"
    return config


if __name__ == "__main__":
  baseline_checkpoint = "/gdrive/MyDrive/Projects/TFM/outputs-dcn/checkpoint.pth"
  no_headh_cp = Path("bacili_detection/detr/detr-r50_no-class-head.pth").resolve()
  DATA_PATH = Path("/gdrive/MyDrive/Projects/TFM/data")
  DB_PATH = DATA_PATH / 'annotations.db'
  os.environ['DATABASE_URI'] = f"sqlite:///{DB_PATH}"
  for i in range(3):
    step(i, 
      "/gdrive/MyDrive/Projects/TFM/outputs/cl-exp-1", 
      device='cuda', 
      model_checkpoint=str(no_headh_cp), 
      incremental=False,
      image_dir="/gdrive/MyDrive/Projects/TFM"
    )


    