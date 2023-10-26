import monai
import numpy as np
from monai.transforms import (
    Compose,
    AddChannel,
    Spacing,
    ScaleIntensity,
    NormalizeIntensityd,
    RandFlipd,
    RandZoomd,
    RandAffined,
    RandGaussianNoised,   
    ToTensord,
    Resize,
    EnsureChannelFirstd
)

train_transforms = Compose([
    # EnsureChannelFirstd(),
    # Spacing(pixdim=(1.5, 1.5, 2.0), mode='bilinear'),
    ScaleIntensity(),
    # NormalizeIntensityd(),
    Resize((96, 96, 96)),
])


test_transforms = Compose([
    # EnsureChannelFirstd(),
    # Spacing(pixdim=(1.5, 1.5, 2.0), mode='bilinear'),
    ScaleIntensity(),
    # NormalizeIntensityd(),
    Resize((96, 96, 96)),
])