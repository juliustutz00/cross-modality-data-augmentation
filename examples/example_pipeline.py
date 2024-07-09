from torchvision.transforms import v2
from ..transformations import CrossModalityTransformations
from ..utils import load_npy_image, display_image_gray, save_npy_image
from ..utils import tensor_to_numpy, numpy_to_tensor, NumpyToTensorTransform, TensorToNumpyTransform
from ..enums import Input_Modality, Output_Modality

# Transformation Instanz erstellen
cross_modality_transformer = CrossModalityTransformations(
    input_modality=Input_Modality.MRI,
    output_modality=Output_Modality.PET
)

# Compose Transforms
composed_transforms = v2.Compose([
    NumpyToTensorTransform(),
    v2.RandomHorizontalFlip(),
    TensorToNumpyTransform(),
    cross_modality_transformer.transform,
    NumpyToTensorTransform(),
    v2.RandomCrop(100),
    TensorToNumpyTransform()
])

# Beispielbild laden und transformieren
input_image = load_npy_image("folder/folder/filename.npy")
output_image = composed_transforms(input_image)

# Ergebnis anzeigen und speichern
display_image_gray(output_image)
save_npy_image("folder/folder/filename.npy", output_image)