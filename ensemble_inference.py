import torch
import torch.nn as nn
from unet import ResUNet, AttentionUNet

# Define the EnsembleModel
class EnsembleModel(nn.Module):
    def __init__(self, resunet_model, attentionunet_model):
        super(EnsembleModel, self).__init__()
        self.resunet_model = resunet_model
        self.attentionunet_model = attentionunet_model

    def forward(self, x):
        # Forward pass through both models
        output_resunet = self.resunet_model(x)
        output_attentionunet = self.attentionunet_model(x)

        # Combine predictions (you can use any method here, like averaging)
        ensemble_output = (output_resunet + output_attentionunet) / 2.0

        return ensemble_output

in_channels = 3  # Assuming RGB images as input
out_channels = 3  # Assuming binary segmentation (1 channel in the mask)

# Load the best weights for the ResUNet and AttentionUNet models
resunet_model = ResUNet(in_channels, out_channels)
resunet_model.load_state_dict(torch.load('best_resunet_weights.pth'))

attentionunet_model = AttentionUNet(in_channels, out_channels)
attentionunet_model.load_state_dict(torch.load('best_attentionunet_weights.pth'))

# Create the EnsembleModel
ensemble_model = EnsembleModel(resunet_model, attentionunet_model)

# Set the model to evaluation mode
ensemble_model.eval()
torch.save(ensemble_model.state_dict(), 'ensemble.pth')
