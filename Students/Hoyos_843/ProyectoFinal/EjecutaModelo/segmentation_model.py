import torch
import segmentation_models_pytorch as smp
from torch import nn

ENCODER = "timm-efficientnet-b0"
WEIGHTS = "imagenet"

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=1,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        logits = self.backbone(images)
        mse_loss = nn.MSELoss()
        if masks is not None:
            if masks.dim() == 3:
                masks = masks.unsqueeze(0)
            return logits, mse_loss(logits, masks)
        return logits