from enum import Enum

class Input_Modality(Enum):
    any = "any"
    PET = "PET"
    MRI = "MRI"
    CT = "CT"

class Output_Modality(Enum):
    PET = "PET"
    MRI = "MRI"
    CT = "CT"
    custom = "custom"