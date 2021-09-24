import numpy as np
from elf.segmentation.watershed import apply_size_filter
from scipy import ndimage


def set_label_to_bg(segmentation, label, bg=0):
    return np.where(segmentation == label, bg, segmentation)


def size_filter_bg_preserving(segmentation, size_filter):
    if size_filter > 0:
        segmentation += 1
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'),
                                            np.zeros_like(segmentation).astype('float32'),
                                            size_filter=size_filter)
        segmentation -= 1
    return segmentation


def get_largest_object(mask):
    """Returns largest connected components"""
    # ~2x faster than clean_object(obj)
    # relabel connected components
    labels, numb_components = ndimage.label(mask)
    assert numb_components > 0  # assume at least 1 CC
    if numb_components == 1:
        return mask
    else:
        return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1


def remove_disconnected_components(segmentation):
    fg_mask = segmentation != 0
    _, numb_components = ndimage.label(fg_mask)
    fg_mask = get_largest_object(fg_mask)
    segmentation[~fg_mask] = 0
    return segmentation
