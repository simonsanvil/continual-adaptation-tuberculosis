# imports and setup
import os, dotenv, sys, glob
from pathlib import Path

# data management
from annotations.object_detection.dataset import DatasetForObjectDetection, ImageForObjectDetection

from annotations.object_detection.rect import Rect
from annotations import db
from sqlalchemy import func
import pandas as pd
import json

# pytoch / scientific computing
from torch.utils.data import DataLoader
import numpy as np
import torch

# model specfic functions
sys.path.append('bacili_detection/detr') # add detr to path
from bacili_detection.detr.datasets.tb_bacillus import TBBacilliDataset, make_ds_transforms
from bacili_detection.detr.models import detr_inference
from bacili_detection.detr import util as detr_util
from bacili_detection.detr.util.misc import get_args_parser, collate_fn
from bacili_detection.detr.util.losses import build_criterion, compute_losses

# visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
%matplotlib inline

# config
dotenv.load_dotenv()
session = db.get_session(os.getenv("DATABASE_URI"))
