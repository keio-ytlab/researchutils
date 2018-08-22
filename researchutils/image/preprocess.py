import numpy as np


def compute_mean_image(frames):
    frame = frames[0]
    prev_frame = frame
    mean_image = np.zeros(shape=frame.shape, dtype=np.float32)
    N = len(frames)
    for frame in frames:
        if not prev_frame.shape == frame.shape:
            raise ValueError("Given frames does not have same shape. {} != {}".format(
                prev_frame.shape, frame.shape))
        mean_image += frame
        prev_frame = frame
    return mean_image / N
