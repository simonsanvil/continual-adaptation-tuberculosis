from typing import Tuple, Union
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


def merge_rects(
        rects:list, 
        confidences:np.ndarray=None, 
        agg:callable=np.mean,
        buffer:float=0.1
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Merge overlapping bounding boxes aggregating their confidence scores if provided

    Parameters
    ----------
    rects: list
        A list of rects (bounding boxes) in the format (x_min, y_min, x_max, y_max)
    confidences: np.ndarray (optional)
        A numpy array of confidence scores for each bounding box.
        Must be of the same length as rects
    agg: callable (optional)
        A function to aggregate the confidence scores of the overlapping bounding boxes.
        Must take a numpy array as input and return a single value. Default is np.mean

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Returns the merged rects and the aggregated confidences if confidences is not None
        Else returns only the merged rects
    """
    from shapely import geometry, ops
    boxes = [geometry.box(*rect) for rect in rects]
    # make union of overlapping boxes
    merged_poly =  ops.unary_union(boxes)
    merged_poly = merged_poly.buffer(buffer) # unit is in pixels 
    if isinstance(merged_poly, geometry.polygon.Polygon) or not hasattr(merged_poly, "geoms"):
        # in case there is only one box
        merged_poly = geometry.MultiPolygon([merged_poly])
    # get the bounding boxes of the merged polygons
    merged_rects = np.array([box.bounds for box in merged_poly.geoms]).astype(int)
    if confidences is not None:
        # identify the boxes that are overlapping to aggregate their confidence scores
        merged_confidences = np.zeros(len(merged_poly.geoms))
        for i, multgeo in enumerate(merged_poly.geoms):
            ind_intersections = [j for j,geo in enumerate(boxes) if geo.intersects(multgeo)]
            merged_confidences[i] = agg(confidences[ind_intersections])
        # return the merged rects and their respective confidences
        return merged_rects, merged_confidences
    else:
        # only return the merged rects
        return merged_rects
    

    