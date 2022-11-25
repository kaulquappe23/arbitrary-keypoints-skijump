
import numpy as np

def get_intermediate_point(start, end, percentage=0.5):
    if start.shape == 3 and (start[2] == 0 or end[2] == 0):
        return np.zeros(3)
    length = np.sqrt(np.sum((start - end) ** 2))
    if length == 0:
        return start
    anno_len = percentage * length
    vec = (start - end) / length
    new_annotation = start - anno_len * vec

    # alternative = percentage * end + (1 - percentage) * start
    # assert np.sqrt(np.sum((new_annotation - alternative) ** 2)) < 1e-5
    return new_annotation


def get_mixed_point(points, percentages):

    assert np.sum(np.asarray(percentages)) - 1 < 1e-8
    new_annotation = np.zeros(3)
    for i, point in enumerate(points):
        if point[2] == 0:
            return np.zeros(3)
        new_annotation += percentages[i] * point

    return new_annotation