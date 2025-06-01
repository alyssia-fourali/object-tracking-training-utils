
""" This module exposes the image processing methods as module level global
variables, that are loaded based on user defined flags.

Since there are multiple different implementations and libraries available for each
of the image tasks we want to perform (e.g 'load jpeg' and 'resize'), we give the
user the choice of using different versions of each function without changing
the rest of the code. One example in which one would use a 'safe' flag is if
there are non jpeg images in the dataset, as imageio.imread can read them,
but not jpeg4py.

The basic changes introduced here are the libjpeg-turbo which is a instruction
level parallel (SSE, AVX, etc) implementation of the default C++ jpeg lib
libjpeg. Morover we also switched the default Pillow for a fork called PIL-SIMD,
which is identical in terms of the API but uses vectorized instructions as well
as heavy loop unrolling to optimize some of Pillows functions (particularly the
resize function). Since the PIL-SIMD is a drop-in replacement for Pillow, even
if the SIMD version is not installed the program should work just as fine with
the default Pillow (without the massive speed gains, though).
"""

from __future__ import absolute_import, division

import torch.nn as nn
import cv2
from math import floor
import numpy as np

from skimage.transform import resize
import PIL
from PIL import Image
try:
    import jpeg4py as jpeg
except (KeyboardInterrupt, EOFError):
    raise
except Exception as e:
    print('[IMAGE-UTILS] package jpeg4py not available. Continuing...')
    LIBJPEG_TURBO_PRESENT = False
else:
    LIBJPEG_TURBO_PRESENT = True

from object_tracking_training_utils.utils.exceptions import InvalidOption

VALID_FLAGS = ['fast', 'safe']
PIL_FLAGS = {'bilinear': PIL.Image.BILINEAR, 'bicubic': PIL.Image.BICUBIC,
             'nearest': PIL.Image.NEAREST}


def decode_jpeg_fast(img_path):
    """ 

    Decodes a JPEG image from the specified file path using the jpeg4py library.

    This method leverages Jpeg decoding method implemented by jpeg4py, available in
    https://github.com/ajkxyz/jpeg4py . This library binds the libjpeg-turbo
    C++ library (available in
    https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/BUILDING.md),
    and can be up to 9 times faster than the non SIMD implementation.
    Requires libjpeg-turbo, built and installed correctly.

    Args:
        img_path (str): The file path to the JPEG image to be decoded.
    Returns:
        numpy.ndarray: The decoded image as a NumPy array in the format (H, W, C).
    Raises:
        FileNotFoundError: If the specified image path does not exist.
        RuntimeError: If the decoding process fails (e.g., corrupted image).
    """
    return jpeg.JPEG(img_path).decode()


def get_decode_jpeg_fcn(flag='fast'):
    """
    Returns the appropriate JPEG decoding function based on the specified flag.

    This method yields the function demanded by the user based on the flags given and
    the system responses to imports. If the demanded function is not
    available an exception is raised and the user is informed it should try
    using another flag.

    Args:
        flag (str): The desired decoding mode. Must be either:
            - 'fast': Uses `decode_jpeg_fast` (requires libjpeg-turbo).
            - 'safe': Uses `imageio.imread` for standard decoding.
            Default is 'fast'.
    Returns:
        function: The selected decoding function (`decode_jpeg_fast` or `imageio.imread`).
    Raises:
        AssertionError: If the flag is set to 'fast' but `libjpeg-turbo` is not available.
        InvalidOption: If the provided flag is not valid. Valid options are:
                       - 'fast'
                       - 'safe'
    """
    if flag == 'fast':
        assert LIBJPEG_TURBO_PRESENT, ('[IMAGE-UTILS] Error: It seems that the '
                                       'used image utils flag is not available,'
                                       ' try setting the flag to \'safe\'.')
        decode_jpeg_fcn = decode_jpeg_fast
    elif flag == 'safe':
        from imageio import imread
        decode_jpeg_fcn = imread
    else:
        raise InvalidOption('The informed flag: {}, is not valid. Valid flags '
                            "include: {}".format(flag, VALID_FLAGS['decode_jpeg']))

    return decode_jpeg_fcn


def resize_fast(img, size_tup, interp='bilinear'):
    """ Implements the PIL resize method from a numpy image input, using the
    same interface as the scipy imresize method.
    OBS: if concerned with the resizing of the correlation score map, the
    default behavior of the PIL resize is to align the corners, so we can
    simply use resize(img, (129,129)) without any problems. The center pixel
    is kept in the center (at least for the 4x upscale) and the corner pixels
    aligned.

    Args:
        img(numpy.ndarray): A numpy RGB image.
        size_tup(tuple): A 2D tuple containing the height and weight of the
            resized image.
        interp(str): The flag indicating the interpolation method. Available
            methods include 'bilinear', 'bicubic' and 'nearest'.
    Returns:
        numpy.ndarray: The resized image.
    """
    # The order of the size tuple is inverted in PIL compared to scipy
    size_tup = (size_tup[1], size_tup[0])
    original_type = img.dtype
    img_res = Image.fromarray(img.astype('uint8', copy=False), 'RGB')
    img_res = img_res.resize(size_tup, PIL_FLAGS[interp])
    img_res = np.asarray(img_res)
    img_res = img_res.astype(original_type)
    return img_res


def get_resize_fcn(flag='fast'):
    """
    Returns the appropriate image resizing function based on the specified flag.

    This method yields the resize function demanded by the user based on the flags given
    and the system responses to imports. If the demanded function is not
    available an exception is raised and the user is informed it should try
    using another flag.

    Args:
        flag (str): The desired resizing mode. Must be either:
            - 'fast': Uses `resize_fast`, optimized for speed.
            - 'safe': Uses `resize`, optimized for compatibility.
            Default is 'fast'.

    Returns:
        function: The selected resizing function (`resize_fast` or `resize`).

    Raises:
        InvalidOption: If the provided flag is not valid. Valid options are:
                       - 'fast'
                       - 'safe'
    """
    if flag == 'fast':
        return resize_fast
    elif flag == 'safe':
        return resize
    else:
        raise InvalidOption('The informed flag: {}, is not valid. Valid flags '
                            "include: {}".format(flag, VALID_FLAGS['resize']))


def pad_image():
    """ Factory function to generate pads dictionaries with the placeholders for
    the amount of padding in each direction of an image. Note: the 'up' direction
    is up in relation to the displayed image, thus it represent the negative
    direction of the y indices.

    Return:
        dictionary: The padding in each direction.
    """
    pads = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    return pads


def crop_img(img, cy, cx, reg_s):
    """ Crops an image given its center and the side of the crop region.
    If the crop exceeds the size of the image, it calculates how much padding
    it should have in each one of the four sides. In the case that the bounding
    box has an even number of pixels in either dimension we approach the center
    of the bounding box as the floor in each dimension, which may displace the
    actual center half a pixel in each dimension. This way we always deal with
    bounding boxes of odd size in each dimension. Furthermore, we add the same
    amount of context region in every direction, which makes the effective
    region size always an odd integer as well, so to enforce this approach we
    require the region size to be an odd integer.

    Args:
        img(numpy.ndarray): The image to be cropped. Must be of dimensions
            [height, width, channels]
        cy(int): The y coordinate of the center of the target.
        cx(int): The x coordinate of the center of the target.
        reg_s(int): The side of the square region around the target, in pixels. Must
            be an odd integer.
    Returns:
        numpy.ndarray: The cropped image.
        dictionary: A dictionary with the amount of padding in pixels in
            each side. The order is: up, down, left, right, or it can be accessed by their names, e.g.: pads['up']
    """
    assert reg_s % 2 != 0, "The region side must be an odd integer."
    pads = pad_image()
    h, w, _ = img.shape
    context = (reg_s-1)/2  # The amount added in each direction
    xcrop_min = int(floor(cx) - context)
    xcrop_max = int(floor(cx) + context)
    ycrop_min = int(floor(cy) - context)
    ycrop_max = int(floor(cy) + context)
    # Check if any of the corners exceeds the boundaries of the image.
    if xcrop_min < 0:
        pads['left'] = -xcrop_min
        xcrop_min = 0
    if ycrop_min < 0:
        pads['up'] = -ycrop_min
        ycrop_min = 0
    if xcrop_max >= w:
        pads['right'] = xcrop_max - w + 1
        xcrop_max = w - 1
    if ycrop_max >= h:
        pads['down'] = ycrop_max - h + 1
        ycrop_max = h - 1
    cropped_img = img[ycrop_min:(ycrop_max+1), xcrop_min:(xcrop_max+1)]

    return cropped_img, pads


def resize_and_pad(cropped_img, out_sz, pads, reg_s=None, use_avg=True, resize_fcn=resize_fast):
    """ Resizes and pads the cropped image.

    Args:
        cropped_img(numpy.ndarray): The cropped image
        out_sz(int): Output size, the desired size of the output
        pads(dictionary): A dictionary of the amount of pad in each side of the
            cropped image. They can be accessed by their names, e.g.: pads['up']
        reg_s(int): Optional: The region side, used to check that the crop size
            plus the padding amount is equal to the region side in each axis
        use_avg(bool): Indicates the mode of padding employed. If True, the
            image is padded with the mean value, else it is padded with zeroes
        resize_fcn: The image resize function.
    Returns:
        numpy.ndarray: The output image, resized and padded. Has size
            (out_sz, out_sz).
    """
    # crop height and width
    cr_h, cr_w, _ = cropped_img.shape
    if reg_s:
        assert ((cr_h+pads['up']+pads['down'] == reg_s) and
                (cr_w+pads['left']+pads['right'] == reg_s)), (
            'The informed crop dimensions and pad amounts are not consistent '
            'with the informed region side. Cropped img shape: {}, Pads: {}, '
            'Region size: {}.'
            .format(cropped_img.shape, pads, reg_s))
    # Resize ratio. Here we assume the region is always a square. Obs: The sum
    # below is equal to reg_s.
    rz_ratio = out_sz/(cr_h + pads['up'] + pads['down'])
    rz_cr_h = round(rz_ratio*cr_h)
    rz_cr_w = round(rz_ratio*cr_w)
    # Resizes the paddings as well, always guaranteeing that the sum of the
    # padding amounts with the crop sizes is equal to the output size in each
    # dimension.
    pads['up'] = round(rz_ratio*pads['up'])
    pads['down'] = out_sz - (rz_cr_h + pads['up'])
    pads['left'] = round(rz_ratio*pads['left'])
    pads['right'] = out_sz - (rz_cr_w + pads['left'])
    # Notice that this resized crop is not necessarily a square.
    rz_crop = resize_fcn(cropped_img, (rz_cr_h, rz_cr_w), interp='bilinear')
    # Differently from the paper here we are using the mean of all channels
    # not on each channel. It might be a problem, but the solution might add a lot
    # of overhead
    if use_avg:
        const = np.mean(cropped_img)
    else:
        const = 0
    # Pads only if necessary, i.e., checks if all pad amounts are zero.
    if not all(p == 0 for p in pads.values()):
        out_img = np.pad(rz_crop, ((pads['up'], pads['down']),
                                   (pads['left'], pads['right']),
                                   (0, 0)),
                         mode='constant',
                         constant_values=const)
    else:
        out_img = rz_crop
    return out_img


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    """
    Crops a square region from the image centered at the specified coordinates 
    and resizes it to the desired output size. If the region exceeds the image 
    boundaries, padding is applied with the specified border type and color.

    Args:
        img (numpy.ndarray): The input image to be cropped and resized.
        center (tuple): (x, y) coordinates of the center of the cropping region.
        size (int): The size of the square region to crop around the center.
        out_size (int): The size of the output cropped region after resizing.
        border_type (int, optional): The type of border to add if padding is required.
            Default is `cv2.BORDER_CONSTANT`.
        border_value (tuple, optional): RGB color to use for padding if BORDER_CONSTANT is used.
            Default is black (0, 0, 0).
        interp (int, optional): Interpolation method for resizing.
            Default is `cv2.INTER_LINEAR`.

    Returns:
        numpy.ndarray: The cropped and resized patch of the image.
    """
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch


def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, use_avg=True):
    """ Pads a frame equally on all sides, enough to fit a region of size
    patch_sz centered in (pos_x, pos_y). If the region is already inside the
    frame, it doesn't do anything.

    Args:
        im(numpy.ndarray): The image to be padded
        frame_sz(tuple): The width and height of the frame in pixels.
        pos_x(int): The x coordinate of the center of the target in the frame.
        pos_y(int): The y coordinate of the center of the target in the frame.
        path_sz(int): The size of the patch corresponding to the context
            region around the bounding box.
        use_avg(bool): Indicates if we should pad with the mean value of the
            pixels in the image (True) or zero (False).
    Returns:
        numpy.ndarray: The image after the padding.
        int: The amount of padding applied.
    """
    c = patch_sz / 2
    xleft_pad = np.maximum(0, - np.round(pos_x - c))
    ytop_pad = np.maximum(0, - np.round(pos_y - c))
    xright_pad = np.maximum(0, np.round(pos_x + c) - frame_sz[1])
    ybottom_pad = np.maximum(0, np.round(pos_y + c) - frame_sz[0])
    npad = np.amax(np.asarray([xleft_pad, ytop_pad, xright_pad, ybottom_pad]))
    npad = np.int32(np.round(npad))
    paddings = ((npad, npad), (npad,npad), (0,0))
    if use_avg:
        im0 = np.pad(im[:, :, 0], paddings[0:2], mode='constant',
                     constant_values=im[:, :, 0].mean())
        im1 = np.pad(im[:, :, 1], paddings[0:2], mode='constant',
                     constant_values=im[:, :, 1].mean())
        im2 = np.pad(im[:, :, 2], paddings[0:2], mode='constant',
                     constant_values=im[:, :, 2].mean())
        im_padded = np.stack([im0,im1,im2], axis=2)
    else:
        im_padded = np.pad(im, paddings, mode='constant')
    return im_padded, npad


def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    """ Extracts the reference patch from the image.

    Args:
        im(numpy.ndarray): The padded image.
        npad(int): The amount of padding added to each side.
        pos_x(int): The x coordinate of the center of the reference in the
            original frame, not considering the padding.
        pos_y(int): The y coordinate of the center of the reference in the
            original frame, not considering the padding.
        sz_src(int): The original size of the reference patch.
        sz_dst(int): The final size of the patch (usually 127).
    Returns:
        numpy.ndarray: The cropped image containing the reference with its context region.
    """
    dist_to_side = sz_src / 2
    # get top-left corner of bbox and consider padding
    tf_x = np.int32(npad + np.round(pos_x - dist_to_side))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tf_y = np.int32(npad + np.round(pos_y - dist_to_side))
    width = np.int32(np.round(pos_x + dist_to_side) - np.round(pos_x - dist_to_side))
    height = np.int32(np.round(pos_y + dist_to_side) - np.round(pos_y - dist_to_side))
    crop = im[tf_y:(tf_y + height), tf_x:(tf_x + width), :]
    crop = imresize(crop, (sz_dst, sz_dst), interp='bilinear')

    # TODO Add Batch dimension
    # crops = np.stack([crop, crop, crop])
    return crop


def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    """ Extracts the 3 scaled crops of the search patch from the image.

    Args:
        im(numpy.ndarray): The padded image.
        npad(int): The amount of padding added to each side.
        pos_x(int): The x coordinate of the center of the bounding box in the
            original frame, not considering the padding.
        pos_y(int): The y coordinate of the center of the bounding box in the
            original frame, not considering the padding.
        sz_src0(int): The downscaled size of the search region.
        sz_src1(int): The original size of the search region.
        sz_src2(int): The upscaled size of the search region.
        sz_dst(int): The final size for each crop (usually 255)
    Returns:
        numpy.ndarray: The 3 cropped images containing the search region in 3 different scales.
    """
    # take center of the biggest scaled source patch
    dist_to_side = sz_src2 / 2
    # get top-left corner of bbox and consider padding
    tf_x = npad + np.int32(np.round(pos_x - dist_to_side))
    tf_y = npad + np.int32(np.round(pos_y - dist_to_side))
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = np.int32(np.round(pos_x + dist_to_side) - np.round(pos_x - dist_to_side))
    height = np.int32(np.round(pos_y + dist_to_side) - np.round(pos_y - dist_to_side))
    search_area = im[tf_y:(tf_y + height), tf_x:(tf_x + width), :]

    # TODO: Use computed width and height here?
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2

    crop_s0 = search_area[np.int32(offset_s0):np.int32(offset_s0 + sz_src0),
                          np.int32(offset_s0):np.int32(offset_s0 + sz_src0), :]
    crop_s0 = imresize(crop_s0, (sz_dst, sz_dst), interp='bilinear')

    crop_s1 = search_area[np.int32(offset_s1):np.int32(offset_s1 + sz_src1),
                          np.int32(offset_s1):np.int32(offset_s1 + sz_src1), :]
    crop_s1 = imresize(crop_s1, (sz_dst, sz_dst), interp='bilinear')

    crop_s2 = imresize(search_area, (sz_dst, sz_dst), interp='bilinear')

    crops = np.stack([crop_s0, crop_s1, crop_s2])
    return crops


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    """
    Reads an image from the specified file path.

    Args:
        img_file(str): The path to the image file.
        cvt_code(int): OpenCV color conversion code.
            Default is `cv2.COLOR_BGR2RGB`, which converts BGR to RGB.
            Use `None` if no conversion is required.
    Returns:
        numpy.ndarray: The image read and optionally converted.
    """
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    """
    Displays an image with optional bounding boxes. The image is resized if its
    dimensions exceed 960 pixels. Bounding boxes can be drawn in either 'ltwh'
    (left, top, width, height) or 'ltrb' (left, top, right, bottom) format.

    Args:
        img(numpy.ndarray): The image to display.
        boxes(list or numpy.ndarray, optional): List of bounding boxes to draw.
        box_fmt(str, optional): Format of bounding boxes, 'ltwh' or 'ltrb'. Default is 'ltwh'.
        colors(list or numpy.ndarray, optional): List of colors for each box.
        thickness(int, optional): Thickness of the bounding box lines.
        fig_n(int, optional): Window number for OpenCV display.
        delay(int, optional): Delay in milliseconds for `cv2.waitKey`. Default is 1 ms.
        visualize(bool, optional): If True, the image is displayed with OpenCV.
        cvt_code(int, optional): Color conversion code. Default is RGB to BGR.
    Returns:
        numpy.ndarray: The image with bounding boxes drawn.
    """
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img
