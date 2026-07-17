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

def diff_images_(
    image1_rgba,
    image2_rgba,
    threshold=0,
    color_space_name='RGB',
    colormap_name='JET',
    color_channels=None
):
    """
    Compare two in-memory images and return:
      1. a grayscale absolute-difference image
      2. a histogram-equalized grayscale difference image
    """

    if image1_rgba.shape != image2_rgba.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"{image1_rgba.shape} vs {image2_rgba.shape}"
        )

    if image1_rgba.dtype != np.uint8:
        image1_rgba = np.clip(image1_rgba, 0, 255).astype(np.uint8)

    if image2_rgba.dtype != np.uint8:
        image2_rgba = np.clip(image2_rgba, 0, 255).astype(np.uint8)

    # Remove alpha and convert explicitly to RGB.
    if image1_rgba.shape[-1] == 4:
        image1_rgb = cv2.cvtColor(
            image1_rgba,
            cv2.COLOR_RGBA2RGB
        )
    elif image1_rgba.shape[-1] == 3:
        image1_rgb = image1_rgba
    else:
        raise ValueError(
            f"Unsupported first image shape: {image1_rgba.shape}"
        )

    if image2_rgba.shape[-1] == 4:
        image2_rgb = cv2.cvtColor(
            image2_rgba,
            cv2.COLOR_RGBA2RGB
        )
    elif image2_rgba.shape[-1] == 3:
        image2_rgb = image2_rgba
    else:
        raise ValueError(
            f"Unsupported second image shape: {image2_rgba.shape}"
        )

    color_space_name = color_space_name.upper()

    if color_space_name == 'RGB':
        image1 = image1_rgb
        image2 = image2_rgb

    elif color_space_name == 'LAB':
        image1 = cv2.cvtColor(
            image1_rgb,
            cv2.COLOR_RGB2LAB
        )
        image2 = cv2.cvtColor(
            image2_rgb,
            cv2.COLOR_RGB2LAB
        )

    elif color_space_name == 'HSV':
        image1 = cv2.cvtColor(
            image1_rgb,
            cv2.COLOR_RGB2HSV
        )
        image2 = cv2.cvtColor(
            image2_rgb,
            cv2.COLOR_RGB2HSV
        )

    else:
        raise ValueError(
            f"Unsupported color space: {color_space_name}"
        )

    if color_channels is not None:
        channels1 = cv2.split(image1)
        channels2 = cv2.split(image2)

        if isinstance(color_channels, int):
            selected_channels = [color_channels]
        else:
            selected_channels = list(color_channels)

        channel_differences = []

        for channel_index in selected_channels:
            channel_differences.append(
                cv2.absdiff(
                    channels1[channel_index],
                    channels2[channel_index]
                )
            )

        if len(channel_differences) == 1:
            difference = channel_differences[0]
        else:
            difference = np.mean(
                np.stack(channel_differences, axis=-1),
                axis=-1
            ).astype(np.uint8)

    elif color_space_name == 'HSV':
        # Hue is circular in OpenCV and ranges from 0 to 179.
        h1, s1, v1 = cv2.split(image1)
        h2, s2, v2 = cv2.split(image2)

        hue_difference = cv2.absdiff(h1, h2)
        hue_difference = np.minimum(
            hue_difference,
            180 - hue_difference
        )

        saturation_difference = cv2.absdiff(s1, s2)
        value_difference = cv2.absdiff(v1, v2)

        difference = np.mean(
            np.stack(
                [
                    hue_difference,
                    saturation_difference,
                    value_difference
                ],
                axis=-1
            ).astype(np.float32),
            axis=-1
        ).astype(np.uint8)

    else:
        difference_rgb = cv2.absdiff(image1, image2)

        if color_space_name == 'RGB':
            difference = cv2.cvtColor(
                difference_rgb,
                cv2.COLOR_RGB2GRAY
            )
        else:
            # LAB channels are not RGB channels, so do not use
            # COLOR_BGR2GRAY or COLOR_RGB2GRAY here.
            difference = np.mean(
                difference_rgb.astype(np.float32),
                axis=-1
            ).astype(np.uint8)

    _, thresholded_diff = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_TOZERO
    )

    equalized_diff = cv2.equalizeHist(
        thresholded_diff
    )

    return thresholded_diff, equalized_diff
