from cv2 import resize
import numpy as np


def preprocess(image, input_shape):
    image = resize(image, tuple(input_shape))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)
