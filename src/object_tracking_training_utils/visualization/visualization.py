import os
import numpy as np

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_frame(frame, bbox, fig_n, pause=2):
    """
    Displays a single video frame with a bounding box overlaid.

    This function opens or updates a matplotlib figure window to display the provided
    image (frame) along with a red rectangle representing the bounding box. It is useful
    for visualizing object tracking or detection results during processing or debugging.

    Args:
        frame (np.ndarray): The image to display, typically a frame from a video.
                            Should be in RGB format with dtype uint8 or convertible to it.
        bbox (list or tuple): Bounding box to draw on the image, in the format [x, y, width, height].
        fig_n (int): The figure number used by matplotlib. This allows reusing the same window.
        pause (float, optional): Duration in seconds to keep the image displayed. Defaults to 2.
    """
    plt.ion()
    plt.clf()
    fig = plt.figure(fig_n)
    ax = fig.gca()
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    fig.show()
    fig.canvas.draw()
    plt.pause(pause)


def show_frame_and_response_map(frame, bbox, fig_n, crop_x, score, pause=2):
    """
    Displays a visualization of the tracking process with three side-by-side views.

    This function is used in the context of object tracking to visually understand
    how the model processes the input and where it believes the tracked object is.
    It shows the current frame with the ground truth bounding box, the reference
    crop used by the tracker, and the response map (heatmap) indicating the predicted
    object location.

    Args:
        frame (np.ndarray): The current video frame (RGB) where the object is being tracked.
        bbox (list or tuple): Ground truth bounding box in the frame, given as [x, y, width, height].
        fig_n (int): The matplotlib figure number to use (helps reuse the same window).
        crop_x (np.ndarray): The template or context crop extracted from the previous frame.
        score (np.ndarray): The response map (heatmap) showing where the tracker predicts the object is.
        pause (float, optional): How long (in seconds) to pause and display the figure. Defaults to 2.
    """
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(131)
    ax.set_title('Tracked sequence')
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Context region')
    ax2.imshow(np.uint8(crop_x))
    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('center')
    ax2.spines['top'].set_color('none')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(133)
    ax3.set_title('Response map')
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['top'].set_color('none')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.imshow(np.uint8(score))

    plt.ion()
    plt.show()
    plt.pause(pause)
    plt.clf()


def save_frame_and_response_map(frame, bbox, fig_n, crop_x, score, writer, fig):
    """
    Saves a visual representation of the tracking process into a video frame.

    The function displays three elements in one figure:
    - The current frame with a bounding box showing the object's position,
    - The reference crop extracted from the previous frame,
    - The response map showing where the model predicts the object is.

    The figure is added as a frame to a video using the provided writer.

    Args:
        frame (np.ndarray): The current RGB video frame.
        bbox (list or tuple): Bounding box coordinates [x, y, width, height].
        fig_n (int): Figure number (not used).
        crop_x (np.ndarray): The cropped region around the object (template).
        score (np.ndarray): The model's response map (prediction heatmap).
        writer: A matplotlib writer object to record video frames.
        fig: A matplotlib figure object to draw the visuals.
    """
    # fig = plt.figure(fig_n)
    plt.clf()
    ax = fig.add_subplot(131)
    ax.set_title('Tracked sequence')
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Context region')
    ax2.imshow(np.uint8(crop_x))
    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('center')
    ax2.spines['top'].set_color('none')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(133)
    ax3.set_title('Response map')
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['top'].set_color('none')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.imshow(np.uint8(score))

    # ax3.grid()
    writer.grab_frame()


def show_crops(crops, fig_n):
    """
    Displays three cropped image patches side by side in a single figure.

    This function is typically used in object tracking tasks to visualize and compare
    different image crops, such as the template, search region, and prediction patch.

    Args:
        crops (np.ndarray): A NumPy array of shape (3, H, W, C), containing three image crops.
                            Each crop should be an RGB image with values convertible to uint8.
        fig_n (int): The matplotlib figure number to use for displaying the crops.
    """
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    """
    Displays three score maps (heatmaps) side by side in a single figure.

    This function is typically used to visualize response maps or confidence scores
    produced by an object tracking or detection model. Each map is displayed using
    a 'hot' colormap to highlight high-confidence areas.

    Args:
        scores (np.ndarray): A NumPy array of shape (3, H, W), containing three 2D score maps.
        fig_n (int): The matplotlib figure number to use for displaying the heatmaps.
    """
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)
