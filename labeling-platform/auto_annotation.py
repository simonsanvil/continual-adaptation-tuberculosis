import os
from typing import List
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import requests, json
import numpy as np
from streamlit_img_label.annotation import read_xml
from annotations.object_detection.rect import Rect, Rects

def get_rects_from_model(serving_uri:str, image:np.ndarray, request_params:dict=None,  **rects_kwargs):
    """
    Annotation the image by requesting the bounding box predictions to a model
    that is deployed on a serving platform.

    Parameters
    ----------
    serving_uri : str
        The URI of the serving platform.
    image : np.ndarray
        The image to be annotated.
    **kwargs : dict
        Additional parameters to be passed to the serving platform.

    Returns
    -------
    predictions : np.ndarray
        The bounding box predictions.
    """
    request_params = request_params or {}
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"data": np.array(image).tolist()})
    resp = requests.post(serving_uri, data=data, headers=headers, params=request_params)
    resp.raise_for_status()
    resp_data = resp.json()
    if "error" in resp_data:
        raise Exception(resp_data["error"])
    else:
        predictions = np.array(resp_data["bboxes"])
    rects = []
    for xmin, ymin, xmax, ymax in predictions.tolist():
        rects.append(
            {
                "left": xmin,
                "top": ymin,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "meta": rects_kwargs.copy(),
            }
        )
    return Rects(rects=[Rect(**rect) for rect in rects])

def save_annotations(annotations_dir, img_path, rects) -> None:
    """
    Save the auto-annotations to an XML file with the same name as the image.

    Parameters
    ----------
    img_path : str
        The path to the image file.
    bounding_boxes : np.ndarray
        The bounding boxes of the image.
    """
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    xml_file = os.path.join(annotations_dir, img_name + ".xml")
    # get the existing annotations if any
    if os.path.exists(xml_file):
        # print(f"Found existing annotations for {img_path}. Merging with auto-annotations.")
        existing_rects = read_xml(xml_file)
        if existing_rects:
            print(f"Found {len(existing_rects)} existing annotations for {img_path}.")
            rects = existing_rects + rects
    if not rects:
        return
    # remove duplicate annotations by merging overlapping bounding boxes
    # rects = merge_rects(rects)
    print(f"Saving {len(rects)} annotations for {img_path}.")
    # save the merged annotations
    with open(xml_file, "w") as f:
        f.write(bbox_to_xml(rects, img_path))

def merge_rects(rects:List[dict]):
    from itertools import groupby
    from shapely.geometry import box
    from shapely import unary_union

    merged_rects = []
    for label, group in groupby(rects, key=lambda x: x["label"]):
        group_polys = [box(rect["left"], rect["top"], rect["left"]+rect["width"], rect["top"]+rect["height"]) for rect in group]
        geom_unions = unary_union(group_polys)
        # convert the polygons to bounding boxes
        if geom_unions.geom_type == "Polygon":
            geoms = [geom_unions]
        elif geom_unions.geom_type == "MultiPolygon":
            geoms = [p for p in geom_unions.geoms]
        else:
            raise Exception("Unexpected geometry type: {}".format(geom_unions.geom_type))
        for geom in geoms:
            xmin, ymin, xmax, ymax = geom.bounds
            merged_rects.append(
                {"left" : int(xmin), "top" : int(ymin), "width" : int(xmax-xmin), "height" : int(ymax-ymin), "label" : label}
            )
    return merged_rects


def _prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def bbox_to_xml(rects, img_path, pose='Unspecified', truncated=0, difficult=0):
    # read the image
    from PIL import Image

    img = Image.open(img_path)
    img = np.arra555rt5rt55t9rtra(img)
    # create the XML

    abspath = os.path.abspath(img_path)
    root = Element('annotation')
    SubElement(root, 'filename').text = os.path.basename(abspath)
    SubElement(root, 'folder').text = os.path.dirname(abspath)
    SubElement(root, 'path').text = abspath
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(img.shape[1])
    SubElement(size, 'height').text = str(img.shape[0])
    SubElement(size, 'depth').text = str(img.shape[2])
    for rect_dict in rects:
        obj = SubElement(root, 'object')
        SubElement(obj, 'label').text = rect_dict['label']
        SubElement(obj, 'pose').text = pose
        SubElement(obj, 'truncated').text = str(truncated)
        SubElement(obj, 'difficult').text = str(difficult)
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(rect_dict['left'])
        SubElement(bndbox, 'ymin').text = str(rect_dict['top'])
        SubElement(bndbox, 'xmax').text = str(rect_dict['left'] + rect_dict['width'])
        SubElement(bndbox, 'ymax').text = str(rect_dict['top'] + rect_dict['height'])
    return _prettify(root)