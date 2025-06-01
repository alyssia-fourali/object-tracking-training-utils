import numpy as np
import torch
import cv2
from video_processing_toolkit.transforms.convert_file_format import ConvertFileFormat


class UtilsDaSiamRPN:

    def __init__(self):
        print(" I am inside init function of test file ")

    @staticmethod
    def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):

        """
        Extracts a square patch (subwindow) from the input image centered at a given position, 
        optionally padding the image if the region goes out of bounds, and resizes it to model_sz.

        Args:
            im (np.ndarray): Input RGB image.
            pos (list or tuple): Center position [x, y] of the patch to extract.
            model_sz (int): Desired output size (width and height) of the patch.
            original_sz (int): Original size (width and height) before resizing.
            avg_chans (list): RGB average values to use for padding.
            out_mode (str): Output format, 'torch' to return a PyTorch tensor, or 'np' to return a numpy array.
            new (bool): Unused flag, reserved for future use.

        Returns:
            np.ndarray or torch.Tensor: Extracted and resized patch.
        """
        # example Preventing overflow after zooming in
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2

        # context_xmin/xmax represents the coordinates of the changed template (possibly negative)

        context_x_min = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
        context_x_max = context_x_min + sz - 1
        context_y_min = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
        context_y_max = context_y_min + sz - 1

        # If greater than 0, no pad
        # If it overflows, pad a certain size distance

        left_pad = int(max(0., -context_x_min))
        top_pad = int(max(0., -context_y_min))
        right_pad = int(max(0., context_x_max - im_sz[1] + 1))
        bottom_pad = int(max(0., context_y_max - im_sz[0] + 1))

        # Relabel the template information on the original image after the padding
        context_xmin = context_x_min + left_pad
        context_xmax = context_x_max + left_pad
        context_ymin = context_y_min + top_pad
        context_ymax = context_y_max + top_pad

        # zzp: a more easy speed version
        # Use avg to fill the boundary part
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                             np.uint8)  # 0 is better than 1 initialization
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1),
                                :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original

        return ConvertFileFormat.im_to_torch(im_patch) if out_mode in 'torch' else im_patch

    @staticmethod
    def cxy_wh_2_rect(pos, sz):
        """
        Converts center-size representation of a bounding box to top-left-size format.

        Args:
            pos (list or np.ndarray): Center position [cx, cy].
            sz (list or np.ndarray): Size [w, h].

        Returns:
            np.ndarray: Bounding box in [x, y, w, h] format, where (x, y) is the top-left corner.
        """
        return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])  # 0-index

    @staticmethod
    def rect_2_cxy_wh(rect):
        """
        Converts a top-left-size bounding box to center-size format.

        Args:
            rect (list or np.ndarray): Bounding box in [x, y, w, h] format.

        Returns:
            tuple: (center [cx, cy], size [w, h]) as two numpy arrays.
        """
        return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]), np.array([rect[2], rect[3]])  # 0-index

    @staticmethod
    def get_axis_aligned_bbox(region):
        """
        Converts a polygon region (4 points) to an axis-aligned bounding box with the same center and area.

        Args:
            region (list or np.ndarray): Either:
                - 8-element flat list [x1, y1, ..., x4, y4]
                - or nested list [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]] as returned by some XML parsers.

        Returns:
            tuple: Bounding box in center-size format (cx, cy, w, h), approximating the input polygon.
        """
        try:
            region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                               region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
        except:
            region = np.array(region)
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        return cx, cy, w, h
