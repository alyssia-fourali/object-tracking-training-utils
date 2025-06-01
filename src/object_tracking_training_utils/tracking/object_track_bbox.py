import numpy as np


def region_to_bbox(region, center=True):
    """
    Converts a ground-truth region annotation to a standard bounding box format.

    The input region can be in one of two formats:
    - 4 values: [x, y, width, height] — axis-aligned bounding box.
    - 8 values: [x1, y1, x2, y2, x3, y3, x4, y4] — rotated bounding box defined by corner points.

    This function normalizes both formats to the standard form (cx, cy, w, h), where:
    - cx, cy = center coordinates of the box
    - w, h   = width and height of the box

    Args:
        region (list or np.ndarray): List of 4 or 8 numbers describing the bounding box.
        center (bool): If True, returns (cx, cy, w, h). If False, returns (x, y, w, h),
                       where (x, y) is the top-left corner.

    Returns:
        tuple: Bounding box in the desired format.
    """
    n = len(region)
    assert n in [4, 8], 'GT region format is invalid, should have 4 or 8 entries.'

    if n == 4:
        return rect(region, center)
    else:
        return poly(region, center)


def rect(region, center):
    """
    Processes a 4-value axis-aligned bounding box.

    If center=True, converts the format from (x, y, w, h) to (cx, cy, w, h),
    where (x, y) is the top-left corner, and (cx, cy) is the center.

    If center=False, returns the original (x, y, w, h).

    Args:
        region (list or np.ndarray): List of 4 values [x, y, width, height].
        center (bool): Whether to convert to center-based format.

    Returns:
        tuple: Bounding box in the chosen format.
    """
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
        return cx, cy, w, h
    else:
        return region


def poly(region, center):
    """
    Processes an 8-value rotated bounding box defined by its four corners.

    Calculates an axis-aligned bounding box that approximates the rotated one,
    preserving its center and approximate area. This is useful when the tracking
    algorithm does not support rotated boxes.

    Args:
        region (list or np.ndarray): List of 8 values representing the (x, y) coordinates
                                     of the four corners of the box, in order.
        center (bool): If True, returns (cx, cy, w, h). Otherwise, returns (x, y, w, h)
                       where (x, y) is the top-left corner of the axis-aligned box.

    Returns:
        tuple: Bounding box approximating the original rotated box.
    """
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    a1 = np.linalg.norm(np.array(region[0:2]) - np.array(region[2:4])) * \
         np.linalg.norm(np.array(region[2:4]) - np.array(region[4:6]))
    a2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(a1/a2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx-w/2, cy-h/2, w, h

