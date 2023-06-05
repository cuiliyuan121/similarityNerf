# The following code is taken from the BlenderProc repository
# https://github.com/DLR-RM/BlenderProc/blob/a68786bb45f3442e93f57434623c81471fdc61a2/blenderproc/python/writer/CocoWriterUtility.py
import numpy as np
from typing import Dict, List
from itertools import groupby
from skimage import measure


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, List[int]]:
    """Converts a binary mask to COCOs run-length encoding (RLE) format. Instead of outputting 
    a mask image, you give a list of start pixels and how many pixels after each of those
    starts are included in the mask.
    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :return: Mask in RLE format
    """
    rle: Dict[str, List[int]] = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def rle_to_binary_mask(rle: Dict[str, List[int]]) -> np.ndarray:
    """Converts a COCOs run-length encoding (RLE) to binary mask.
    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts: List[int] = rle.get('counts')

    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


def bbox_from_binary_mask(binary_mask: np.ndarray) -> List[int]:
    """ Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

    :param binary_mask: A binary image mask with the shape [H, W].
    :return: The bounding box represented as [x, y, width, height]
    """
    # Find all columns and rows that contain 1s
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    # Find the min and max col/row index that contain 1s
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Calc height and width
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    return [int(cmin), int(rmin), int(w), int(h)]


def calc_binary_mask_area(binary_mask: np.ndarray) -> int:
    """ Returns the area of the given binary mask which is defined as the number of 1s in the mask.

    :param binary_mask: A binary image mask with the shape [H, W].
    :return: The computed area
    """
    return binary_mask.sum().tolist()


def close_contour(contour: np.ndarray) -> np.ndarray:
    """ Makes sure the given contour is closed.

    :param contour: The contour to close.
    :return: The closed contour.
    """
    # If first != last point => add first point to end of contour to close it
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask: np.ndarray, tolerance: int = 0) -> List[np.ndarray]:
    """Converts a binary mask to COCO polygon representation

     :param binary_mask: a 2D binary numpy array where '1's represent the object
     :param tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If
                       tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = np.array(measure.find_contours(padded_binary_mask, 0.5))
    # Reverse padding
    contours = contours - 1
    for contour in contours:
        # Make sure contour is closed
        contour = close_contour(contour)
        # Approximate contour by polygon
        polygon = measure.approximate_polygon(contour, tolerance)
        # Skip invalid polygons
        if len(polygon) < 3:
            continue
        # Flip xy to yx point representation
        polygon = np.flip(polygon, axis=1)
        # Flatten
        polygon = polygon.ravel()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        polygon[polygon < 0] = 0
        polygons.append(polygon.tolist())

    return polygons
