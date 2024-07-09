import numpy as np
from torch import from_numpy, Tensor

def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def numpy_to_tensor(array: np.ndarray) -> Tensor:
    return from_numpy(array)

class NumpyToTensorTransform:
    def __call__(self, img):
        return numpy_to_tensor(img)

class TensorToNumpyTransform:
    def __call__(self, img):
        return tensor_to_numpy(img)