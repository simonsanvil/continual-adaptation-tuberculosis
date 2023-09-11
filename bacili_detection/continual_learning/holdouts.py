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
dotenv.load_dotenv(".env")
DEFAULT_HOLDOUTS = [
    0.15, 0.25, 0.35, 0.25
]
# LABELS = [
#     "random_sampling_incremental",
#     "random_sampling_retraining",
#     "active_learning_loss_prediction_incremental",
#     "active_learning_loss_prediction_retraining",
#     "active_learning_uncertainty_incremental",
#     "active_learning_uncertainty_retraining",
# ]

def make_holdouts(
        source_tag:str="train",
        holdout_tag:str="holdout",
        frac:float=0.5,
        nonholdout_tag:str="holdout:-1",
        session=None,
        **kwargs
):
    if session is None:
        session = db.get_session(os.environ.get("DATABASE_URI"))
    art_ds = TBBacilliDataset(source_tag, db_session=session, **kwargs)
    artifacts = [imod.artifact for imod in art_ds._images]
    inds = np.arange(len(artifacts))
    # print( len(artifacts) * frac)
    holdout_artifacts_inds = np.random.choice(
        inds, size=int(len(artifacts) * frac), replace=False
    )
    # tag them as holdout
    for i in holdout_artifacts_inds:
        artifact = artifacts[i]
        newtag = db.ArtifactTag(tag=holdout_tag, artifact_id=artifact.id)
        session.add(newtag)
    session.commit()
    # add the tag 'incremental_training' tag to the rest
    for i in inds:
        if i not in holdout_artifacts_inds:
            artifact = artifacts[i]
            newtag = db.ArtifactTag(tag=nonholdout_tag, artifact_id=artifact.id)
            session.add(newtag)
    session.commit()
    print(f"Created {len(holdout_artifacts_inds)} holdouts with tag '{holdout_tag}'")

def make_holdouts_steps(
        *holdouts, 
        session=None,
        holdout_tag:str="holdout",
        tag_suffix:str="",
        project_name:str="Bacilli Detection"
    ):
    """
    Make holdouts in the database
    """
    if session is None:
        session = db.get_session(os.environ.get("DATABASE_URI"))
    
    holdout_artifacts = (
        session.query(db.Artifact)
        .join(db.Project)
        .join(db.ArtifactTag, isouter=True)
        .where(db.Project.name == project_name)
        .where(db.ArtifactTag.tag == holdout_tag)
        .group_by(db.Artifact.id)
        .all()
    )
    print(f"Found {len(holdout_artifacts)} artifacts with tag '{holdout_tag}'")

    # divide the holdouts into the sets according to holdouts list of floats
    if len(holdouts) == 0:
        holdouts = DEFAULT_HOLDOUTS
    else:
        holdouts = list(holdouts)

    assert sum(holdouts) <= 1, f"Sum of holdouts must be less than 1, but is {sum(holdouts)}"

    print(f"Making holdouts with fractions: {holdouts}")
    holdouts_left = set(holdout_artifacts.copy())
    for i, frac in enumerate(holdouts):

        if i == len(holdouts) - 1:
            holdouts_frac = list(holdouts_left)
        else:
            print(f"Making holdout {i+1} of {len(holdouts)} with fraction {frac} of size {int(len(holdout_artifacts) * frac)}")
            holdouts_frac = np.random.choice(list(holdouts_left), size=int(len(holdout_artifacts) * frac), replace=False)
        
        holdouts_left = holdouts_left - set(holdouts_frac)

        tag_name = f"holdout:{i}{tag_suffix}"
        tag_holdouts(holdouts_frac, tag_name, session)

        if len(holdouts_left) == 0:
            break

def query_holdouts_at_step(session, step:int, holdout_name:str="holdout"):
    """
    Query the holdouts with the given name
    """
    return (
        session.query(db.Artifact)
        .join(db.ArtifactTag)
        .where(db.ArtifactTag.tag.like(f"{holdout_name}:{step}%"))
        .all()
    )

def get_all_holdouts(session):
    """
    Get all holdouts from the database
    """
    return (
        session.query(db.Artifact)
        .join(db.ArtifactTag)
        .where(db.ArtifactTag.tag.like("holdout%"))
        .all()
    )

def get_holdout_tags(session, holdout_name:str="holdout", suffix:str="", steps:bool=False):
    """
    Get all holdout tags from the database
    """
    if steps:
        suffix = ":"+suffix
    tags = (
        session.query(db.ArtifactTag.tag)
        .where(db.ArtifactTag.tag.like(f"{holdout_name}{suffix}%"))
        .all()
    )
    return {tag[0] for tag in tags}

def tag_holdouts(holdouts:list, tag_name:str, session):
    """
    Label the holdouts with the given label name
    """
    for i, artifact in enumerate(holdouts):
        tag = db.ArtifactTag(tag=tag_name, artifact_id=artifact.id)
        session.add(tag)
    
    session.commit()
        # print(f"Labeling holdout {i+1} of {len(holdouts)}")

def delete_all_holdouts(session, holdout_name:str="holdout"):
    """
    Delete all holdouts from the database
    """
    session.query(db.ArtifactTag).where(db.ArtifactTag.tag.like(f"{holdout_name}%")).delete()
    session.commit()