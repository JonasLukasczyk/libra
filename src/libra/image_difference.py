"""
This module provides functions to compare two images and generate a thresholded difference image.

It supports various color spaces and allows for comparing specific color channels.
The thresholded difference image highlights the significant differences between the two images.
"""

import cv2

from .utils import *


def diff_images(image1_path, image2_path, threshold=0, color_space_name='HSV', colormap_name='JET', color_channels=None):
    """
    Load two images, convert them to the specified color space, calculate the difference,
    and apply a threshold to highlight significant differences

    Args:
        image1_path (str): The path to the first image file.
        image2_path (str): The path to the second image file.
        threshold (int): The threshold value to apply for highlighting differences.
        color_space_name (str): name of the color space to use
        colormap_name (str, optional) : name of the color map for the difference
        color_channels (tuple or None, optional): A tuple specifying the color channels to use for calculating the difference.
            If None, all color channels are used.

    Raises:
        FileNotFoundError: If either of the input image files is not found.

    Return:
        image difference image (open CV)
        iamge difference histogram equalized (open CV)
    """
    # Load and convert images
    color_space_code = get_color_space_code(color_space_name)
    image1 = load_and_convert_image(image1_path, color_space_code)
    image2 = load_and_convert_image(image2_path, color_space_code)

    colormap = color_maps['JET']
    if colormap_name in color_maps:
        colormap = color_maps[colormap_name]
    else:
        print(f"{colormap_name} colormap does not exist. JET will be used!")


    if color_channels is None:
        # Calculate the absolute difference between the images
        difference = cv2.absdiff(image1, image2)
    else:
        # Calculate the absolute difference between specific color channels
        image1_channels = cv2.split(image1)
        image2_channels = cv2.split(image2)
        difference = cv2.absdiff(image1_channels[color_channels], image2_channels[color_channels])

    # Apply threshold
    _, thresholded_diff = cv2.threshold(difference, threshold, 255, cv2.THRESH_TOZERO)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(thresholded_diff, cv2.COLOR_BGR2GRAY)

    # histogram equalize the difference
    equalized_diff = cv2.equalizeHist(diff_gray)

    # Apply a colormap to visualize the difference
    heatmap = cv2.applyColorMap(diff_gray, colormap)
    heatmapEq = cv2.applyColorMap(equalized_diff, colormap)

    return heatmap, heatmapEq

def diff_images_(image1_rgba, image2_rgba, threshold=0, color_space_name='HSV', colormap_name='JET', color_channels=None):
    """
    Load two images, convert them to the specified color space, calculate the difference,
    and apply a threshold to highlight significant differences

    Args:
        image1_path (str): The path to the first image file.
        image2_path (str): The path to the second image file.
        threshold (int): The threshold value to apply for highlighting differences.
        color_space_name (str): name of the color space to use
        colormap_name (str, optional) : name of the color map for the difference
        color_channels (tuple or None, optional): A tuple specifying the color channels to use for calculating the difference.
            If None, all color channels are used.

    Raises:
        FileNotFoundError: If either of the input image files is not found.

    Return:
        image difference image (open CV)
        iamge difference histogram equalized (open CV)
    """
    # Load and convert images
    color_space_code = get_color_space_code(color_space_name)

    image1 = cv2.cvtColor(image1_rgba, color_space_code)
    image2 = cv2.cvtColor(image2_rgba, color_space_code)

    colormap = color_maps['JET']
    if colormap_name in color_maps:
        colormap = color_maps[colormap_name]
    else:
        print(f"{colormap_name} colormap does not exist. JET will be used!")

    if color_channels is None:
        # Calculate the absolute difference between the images
        difference = cv2.absdiff(image1, image2)
    else:
        # Calculate the absolute difference between specific color channels
        image1_channels = cv2.split(image1)
        image2_channels = cv2.split(image2)
        difference = cv2.absdiff(image1_channels[color_channels], image2_channels[color_channels])

    # Apply threshold
    _, thresholded_diff = cv2.threshold(difference, threshold, 255, cv2.THRESH_TOZERO)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(thresholded_diff, cv2.COLOR_BGR2GRAY)

    # histogram equalize the difference
    equalized_diff = cv2.equalizeHist(diff_gray)

    return diff_gray, equalized_diff


def __dir__():
    return ["diff_images"]
