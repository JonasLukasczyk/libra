from .utils import *
from .metrics import *


def compute_metric(dist_path, ref_path, metric_name='SSIM', color_space_name='LAB'):
    """
    Compute the specified IQA metrics for the given images and color spaces.

    Args:
        dist_path (str): Path of distorted image.
        ref_path (str): Path of reference image.
        metric_name (str): metrics to use e.g. 'SSIM'.
        color_space_name (str): Name of the color spaces to use for computing metrics e.g. 'LAB'

    Returns:
        list: List of dictionaries containing metric results for each color space.
    """
    dist_image = load_image(dist_path)
    ref_image = load_image(ref_path)


    if metric_name in metrics:
        metric_fn = metrics[metric_name]

        color_space_code = color_spaces[color_space_name]
        dist_image_converted = cv2.cvtColor(dist_image, color_space_code)
        ref_image_converted  = cv2.cvtColor(ref_image, color_space_code)

        if metric_name in ["BRISQUE", "NIQE", "MUSIQ", "NIMA", "CLIPIQA"]:
            result = metric_fn(dist_image_converted)
        else:
            result = metric_fn(dist_image_converted, ref_image_converted)

        return result
    else:
        return float("nan")

def compute_metric_(dist_image_rgba, ref_image_rgba, metric_name='SSIM', color_space_name='LAB'):
    dist_image = cv2.cvtColor(dist_image_rgba, cv2.COLOR_RGBA2BGR)
    ref_image = cv2.cvtColor(ref_image_rgba, cv2.COLOR_RGBA2BGR)

    if metric_name in metrics:
        metric_fn = metrics[metric_name]

        color_space_code = color_spaces[color_space_name]
        dist_image_converted = cv2.cvtColor(dist_image, color_space_code)
        ref_image_converted  = cv2.cvtColor(ref_image, color_space_code)

        if metric_name in ["BRISQUE", "NIQE", "MUSIQ", "NIMA", "CLIPIQA"]:
            result = metric_fn(dist_image_converted)
        else:
            result = metric_fn(dist_image_converted, ref_image_converted)

        return result
    else:
        return float("nan")



def list_metrics():
    """
    Returns:
        all the supported metrics
    """
    return metrics.keys()


def list_colorspaces():
    """
    Returns:
        all the supported color spaces
    """
    return color_spaces.keys()


def __dir__():
    return ["compute_metrics", "compute_metric", "compute_metric_", "list_metrics", "list_colorspaces"]
