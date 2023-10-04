from typing import Dict, Tuple, Union, List
import numpy as np
import cv2
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter('%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s')
logger.handlers[0].setFormatter(formatter)

class SputumDetectionModel:

    def __init__(self, model, chunk_size=80, stride=40, verbose:bool=True):
        self.model = model
        self.chunk_size = chunk_size
        self.stride = stride
        self.verbose = verbose
    
    def _predict_raw(self, img, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """predict the class of the image"""
        device = kwargs.pop("device", None)
        img_cuts, coords = self._preprocess(img, **kwargs)
        if device is not None:
            with tf.device(device):
                out = self.model.predict(img_cuts, verbose=self.verbose)
        else:
            out = self.model.predict(img_cuts, verbose=self.verbose)
        return out, coords

    def _preprocess(self, img, *, reversed_img=True):
        """preprocess the image to make it ready to be predicted"""
        scaled_img = self._scale_img(img)/255
        coords, _, _ = self.tile_coords(scaled_img)
        cuts = self._get_chunks(scaled_img, coords, reversed=reversed_img)
        return cuts, coords.astype(np.uint16)
    
    # @keras.utils.register_keras_serializable(package='SputumDetectionModel')
    def _preprocess_fast(self, img):
        """preprocess the image to make it ready to be predicted"""
        scaled_img = self._scale_img(img)
        padded_img = self._pad_img(scaled_img)
        cuts, coords = self.tile_cut(padded_img)
        # remove the cuts that are inside the image
        xmask = coords[:,2] < img.shape[0]
        ymask = coords[:,3] < img.shape[1]
        cuts = cuts[xmask & ymask]
        coords = coords[xmask & ymask]
        return cuts, coords
    
    @staticmethod
    def _scale_img(img):
        min, max = img.min(), img.max()
        img= (((img - min)/(max-min))*255)
        img = img.astype(np.uint8)
        imout = img.copy()
        return imout


    def tile_coords(self, img:np.ndarray, kernel_size:Tuple[int, int]=None, stride: int=None):
        kernel_size = kernel_size or (self.chunk_size, self.chunk_size)
        stride = stride or self.stride
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        else:
            h, w = img[:2]
        num_cols_tiles = (h // stride) + 1
        num_rows_tiles = (w // stride) + 1
        inds = np.arange(num_cols_tiles * num_rows_tiles)
        x = np.array((inds // num_cols_tiles) * stride, dtype=np.uint16)
        y = np.array((inds % num_cols_tiles) * stride, dtype=np.uint16)
        coords = np.stack([x, y, x + kernel_size[0], y + kernel_size[1]], axis=1)
        # move coords outside the image to the edge
        x_mask = coords[:,2] >= w
        y_mask = coords[:,3] >= h
        x_shift = coords[x_mask, [2]] % w
        y_shift = coords[y_mask, [3]] % h
        coords[x_mask,0] = coords[x_mask,0] - x_shift
        coords[x_mask,2] = coords[x_mask,2] - x_shift
        coords[y_mask,1] = coords[y_mask,1] - y_shift
        coords[y_mask,3] = coords[y_mask,3] - y_shift
        return coords, num_rows_tiles, num_cols_tiles
    
    def _get_chunks(self, img, coords, reversed=False, mask=None):
        if mask is not None:
            # get the chunks where the img is not fully black
            # after applying the mask
            masked_img = img * mask.reshape(*mask.shape, 1)
            coords = [
                (x1, y1, x2, y2)
                for x1, y1, x2, y2 in coords
                if masked_img[x1:x2, y1:y2].max() > 0
            ]
        if reversed:
            chunks = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in coords]
        else:
            chunks = [img[x1:x2, y1:y2] for x1, y1, x2, y2 in coords]
        return np.array(chunks)

    def _pad_img(self, img, kernel_size:Tuple[int,int]=None) -> np.ndarray:
        """pad the image to make it divisible by the kernel size"""
        kernel_size = kernel_size or (self.chunk_size, self.chunk_size)
        x_pad = kernel_size[1] - img.shape[1] % kernel_size[1] + 1
        y_pad = kernel_size[0] - img.shape[0] % kernel_size[0] + 1
        padded_img = np.pad(img, ((0,y_pad), (0,x_pad), (0,0)), mode='constant', constant_values=0)
        return padded_img

    def tile_cut(self, img, kernel_size:Tuple[int,int]=None, stride:int=None) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Cut the image in squared tiles of self.tile_size with a stride of self.stride pixels. 
        Returns the image cuts and the coordinates corresponding to the position of the cuts in the original image.
        """
        kernel_size = kernel_size or (self.chunk_size, self.chunk_size)
        stride = stride or self.stride
        # cut the image in tiles
        ashp = np.array(img.shape)
        c = ashp[-1]
        window = (*kernel_size, c)
        steps = (stride, stride, c)
        wshp = np.array(window).reshape(-1)
        if steps:
            stp = np.array(steps).reshape(-1)
        else:
            stp = np.ones_like(ashp)
        astr = np.array(img.strides)
        assert np.all(np.r_[ashp.size == wshp.size, wshp.size == stp.size, wshp <= ashp])
        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)
        # resize img to fit the window size
        aview = np.lib.stride_tricks.as_strided(img, shape = shape, strides = strides)
        # get the coordinates of the cuts
        indices = np.arange(aview.shape[0] * aview.shape[1])
        coords_x = (indices // aview.shape[1]) * stride
        coords_y = (indices % aview.shape[1]) * stride
        coords = np.stack([coords_x, coords_y, coords_x + kernel_size[0], coords_y + kernel_size[1]], axis=1)
        tiles = aview.reshape(-1, *kernel_size, c)
        return tiles, coords
    
    @classmethod
    def from_keras(cls, path, chunk_size=80, stride=40, verbose:bool=True):
        """create a new instance of the class from a keras model"""
        from keras.models import load_model
        model = load_model(path)
        return cls(model, chunk_size, stride, verbose)

    @classmethod
    def visualize_prediction(cls, img, bboxes, conf=None, merge_rects:bool=False, **kwargs):
        """
        Visualize the bounding boxes on the image predicted by the model

        Parameters
        ----------
        img: np.ndarray
            The image to be predicted
        bboxes: np.ndarray
            The bounding boxes predicted by the model
        conf: np.ndarray (optional)
            The confidence of the respective bounding boxes
        merge_rects: bool (optional)
            Whether to merge overlapping bounding boxes prior to plotting or not
        **kwargs
            Additional arguments to be passed to cls.plot_rects (e.g. thresholds, asarray, ax)
        """
        if merge_rects:
            rects = cls.merge_rects(bboxes, conf)
        else:
            rects = bboxes
        print("Objects found: ", len(rects))
        cls.plot_rects(img, rects, **kwargs)

    @classmethod
    def plot_rects(cls, img:np.ndarray, rects:np.ndarray, 
                   conf:np.ndarray=None, thresholds:Dict[float, str]=None, 
                   asarray:bool=False, plot_lower:Union[bool, dict]=True, 
                   ax=None, **kwargs):
        """
        plot the bounding boxes on the image

        Parameters
        ----------
        img: np.ndarray
            The image to be predicted
        rects: np.ndarray
            The bounding boxes to visualize
        conf: np.ndarray (optional)
            The confidence of the respective bounding boxes
        thresholds: dict (optional)
            A dictionary of (threshold, color) pairs to be used to color the bounding boxes according to their confidence.
            The threshold is a float between 0 and 1 representing the confidence of the bounding box.
            The color is a tuple of 3 integers (RGB) between 0 and 255.
            boxes with a confidence in the range [threshold_i, threshold_{i+1}) will be colored with the color of threshold_i
        asarray: bool (optional)
            Whether to return the image as a numpy array or to plot it with matplotlib
        
        Returns
        -------
        None or np.ndarray
            Returns the image with the bounding boxes plotted if asarray is True
        """
        import matplotlib.pyplot as plt
        img = img.copy()
        if thresholds is None:
            # green if 0.5, yellow if 0.7, red if 0.9 confidence
            thresholds = ((0.5, (255, 0, 0)), (0.7, (255, 255, 0)), (0.9, (0, 255, 0)))
        else:
            thresholds = sorted(tuple(thresholds.items()), key=lambda x: x[0])
        if plot_lower is True:
            plot_lower = {"color": (0, 0, 0), "thickness": 1}
        for i, (x_min, y_min, x_max, y_max) in enumerate(rects):
            style_dict_ = {}
            if conf is not None:
                color = [c for t, c in thresholds if conf[i] >= t]
                if len(color) == 0: # no threshold was met, use black
                    if plot_lower:
                        color = plot_lower
                    else:
                        continue
                else:
                    color = color[-1]
            else:
                color = thresholds[-1][1]
            if isinstance(color, dict):
                style_dict_ = color
            else:
                style_dict_ = {'color': color}
            style_dict = {'color': (0, 255, 0), 'thickness': 3, 'lineType': cv2.LINE_AA}
            style_dict.update(kwargs)
            style_dict.update(style_dict_)
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max),int(y_max)), **style_dict)
        if asarray:
            return img
        if ax is None:
            fig, ax = plt.subplots(
                1, figsize=(img.shape[1]/100, img.shape[0]/100),
                dpi=100, tight_layout=True, frameon=False
            )
        # remove white borders
        ax.axis('off')
        fig = ax.get_figure()
        fig.tight_layout(pad=0, w_pad=0, h_pad=0, rect=None)
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        ax.imshow(img)


    @classmethod
    def merge_rects(
            cls, rects:list, confidences:np.ndarray=None, agg:callable=np.mean
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Merge overlapping bounding boxes aggregating their confidence scores.

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
        from shapely import geometry
        boxes = [geometry.box(*rect) for rect in rects]
        # make union of overlapping boxes
        merged_poly = geometry.MultiPolygon(boxes)
        merged_poly = merged_poly.buffer(0.1) # unit is in pixels (in this case 0.1 pixels)
        if isinstance(boxes, geometry.polygon.Polygon) or not hasattr(merged_poly, "geoms"):
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
    
    def predict(
            self, img, th:float=0.5, 
            merge_rects:bool=True,
            confidences=True,
            report_center:bool=True,
            verbose:bool=False,
            **kwargs
        ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        predict the bounding boxes of the nodules in the image.
        Returns a tuple of the detected bounding boxes and their respective 
        confidences when `confidences=True` (default), otherwise, only the bounding boxes
        are returned.

        Parameters
        ----------
        img : np.ndarray
            the image to be predicted
        th : float, optional
            the threshold to be used to filter the bounding boxes, by default 0.5
        merge_rects : bool, optional
            whether to merge overlapping bounding boxes or not, by default True
        confidences : bool, optional
            whether to return the confidences of the bounding boxes or not, by default True

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            the bounding boxes and their respective confidences (if `confidences=True`)
        """
        # logger.info("Predicting bounding boxes")
        out, coords = self._predict_raw(img, **kwargs)
        rects = coords[out[:, 0] > th]
        if report_center:
            offset = (self.chunk_size / 2)//2
            rects = rects + np.array([offset, offset, -offset, -offset])
        if len(rects) == 0:
            logger.info("No bounding boxes were found")
            return rects
        confs = out[out[:, 0] > th][:, 0]
        if verbose:
            logger.info(f"Predicted {len(rects)} potential rects")
        if merge_rects:
            rects, confs = self.merge_rects(rects, confidences=confs)
        if verbose:
            logger.info(f"A total of {len(rects)} bounding boxes were found")
        if confidences:
            return rects, confs
        return rects
    

import mlflow
from typing import List

class SputumDetectorPyfunc(mlflow.pyfunc.PythonModel):

    def __init__(self, **params):
        super().__init__()
        self.params = params
    
    def load_context(self, context):
        from keras.models import load_model
        keras_model = load_model(context.artifacts["model"])
        self.detector = SputumDetectionModel(keras_model, **self.params)

    def predict(self, context, img:List[List[float]]) -> List[List[float]]:
        img = np.array(img, dtype=np.uint8)
        out, chunks = self.detector.predict(img)
        return self.detector.get_bounding_boxes(img, out, chunks)
    
    @classmethod
    def log_keras(cls, model_name, keras_path, pip_requirements=None, **kwargs):     
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts = {"model": keras_path},
            code_path=[__file__],
            pip_requirements=pip_requirements,
        )
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("chunk_size", kwargs.pop("chunk_size", 80))
        mlflow.log_param("stride", kwargs.pop("stride", 40))
        mlflow.log_param("verbose", kwargs.pop("verbose", True))
        for key, value in kwargs.items():
            mlflow.log_param(key, value)