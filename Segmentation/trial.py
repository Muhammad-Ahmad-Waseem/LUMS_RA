import segmentation_models_pytorch as smp
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(r"D:\LUMS_RA\Models\Segmentation\trained_model\DeepLabV3+\Combined_Images\best_model.h5",
                   map_location=DEVICE)