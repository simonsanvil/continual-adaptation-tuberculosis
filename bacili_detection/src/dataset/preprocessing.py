from typing import Tuple
import cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

def mask_filter(img:np.ndarray) -> np.ndarray:
    """
    preprocess image only keeping red-pink colored pixels
    and applying erosion and dilation
    """
    # First stage: convert to hsv and keep only red-pink pixels
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # hsv_blurred = cv2.GaussianBlur(hsv, (7, 7), 0)
    blue_green_mask = cv2.inRange(
        hsv, np.array([45, 0, 0]), np.array([115, 255, 255])
    )
    # all black or all white
    white_mask = cv2.inRange(
        hsv, np.array([0, 0, 200]), np.array([255, 255, 255])
    )
    black_mask = cv2.inRange(
        hsv, np.array([0, 0, 0]), np.array([255, 255, 50])
    )
    mask = blue_green_mask + white_mask + black_mask
    # slice the red
    imask = mask == 0
    return imask

@jit(nopython=True)
def tile_coords(
    width:np.uint16, 
    height:np.uint16, 
    kernel_size:np.uint16=None, 
    stride: np.uint16=None
) -> Tuple[np.ndarray, np.uint16, np.uint16]:
    num_cols_tiles = (height // stride) + 1
    num_rows_tiles = (width // stride) + 1
    inds = np.arange(0, num_cols_tiles * num_rows_tiles, 1, np.uint16)
    x = (inds // num_cols_tiles) * stride
    y = (inds % num_cols_tiles) * stride
    num_inds = num_cols_tiles * num_rows_tiles
    coords = np.zeros((num_inds, 4), np.uint16)
    coords[:,0] = x
    coords[:,1] = y
    coords[:,2] = x + kernel_size
    coords[:,3] = y + kernel_size
    # # move coords outside the image to the edge
    x_mask = coords[:,2] >= width
    y_mask = coords[:,3] >= height
    x_shift = coords[x_mask, 2] % width
    y_shift = coords[y_mask, 3] % height
    coords[x_mask,0] = coords[x_mask,0] - x_shift
    coords[x_mask,2] = coords[x_mask,2] - x_shift
    coords[y_mask,1] = coords[y_mask,1] - y_shift
    coords[y_mask,3] = coords[y_mask,3] - y_shift
    return coords, num_cols_tiles, num_rows_tiles