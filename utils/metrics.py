import numpy as np
from utils.transform import bdb3d_corners


def bdb3d_iou(obj3d1, obj3d2):
    from shapely.geometry.polygon import Polygon
    corners1 = bdb3d_corners(obj3d1)
    corners2 = bdb3d_corners(obj3d2)

    polygon1 = Polygon(corners1[(0, 1, 3, 2), :2])
    polygon2 = Polygon(corners2[(0, 1, 3, 2), :2])
    inter2d = polygon1.intersection(polygon2).area
    inter3d = inter2d * max(0., min(corners1[4, 2], corners2[4, 2]) - max(corners1[0, 2], corners2[0, 2]))

    vol1 = polygon1.area * (corners1[4, 2] - corners1[0, 2])
    vol2 = polygon2.area * (corners2[4, 2] - corners2[0, 2])

    return inter3d / (vol1 + vol2 - inter3d)


def bdb2d_iou(obj2d1, obj2d2):
    cmin1, rmin1, w1, h1 = obj2d1['bdb2d']
    cmax1, rmax1 = cmin1 + w1, rmin1 + h1
    cmin2, rmin2, w2, h2 = obj2d2['bdb2d']
    cmax2, rmax2 = cmin2 + w2, rmin2 + h2

    inter_area = max(0, min(cmax1, cmax2) - max(cmin1, cmin2)) * max(0, min(rmax1, rmax2) - max(rmin1, rmin2))
    area1 = w1 * h1
    area2 = w2 * h2
    iou = inter_area / (area1 + area2 - inter_area)
    assert 0 <= iou <= 1
    return iou


def seg_iou(obj2d1, obj2d2):
    seg1 = obj2d1['segmentation']
    seg2 = obj2d2['segmentation']
    inter = np.logical_and(seg1, seg2).sum()
    union = np.logical_or(seg1, seg2).sum()
    return inter / union
