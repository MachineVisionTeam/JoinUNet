import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
from PIL import Image
import torch.nn.init as init
import os
import time
from unet import ResUNet
from unet import AttentionUNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import StepLR
from torch_lr_finder import TrainDataLoaderIter
import torch.nn.functional as F

torch.manual_seed(0)

# Define a function to load images
def load_image(filename):
    ext = filename.suffix.lower()
    if ext in ['.npy', '.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.png':
        return Image.open(filename)
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir):
    mask_file = mask_dir / f"{idx} .png"
    mask = np.asarray(load_image(mask_file))

    if not mask_file.is_file():
        raise ValueError(f"No mask file found for index: {idx}")

    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class CarvanaCustomDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        # Modify the mask suffix to match the file extension of your mask files
        #self.mask_suffix = '.png'

        self.ids = [file.stem for file in self.images_dir.glob('*.tif') if not file.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

    def __len__(self):
        return len(self.ids)
    def shuffle_data(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.ids = [self.ids[i] for i in indices]

    def preprocess_image(self, image_path):
        # Load the image using Pillow (PIL)
        img = Image.open(image_path)
    

        # Apply your preprocessing steps while preserving the number of channels
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Applies random rotation, translation, and scaling.
            transforms.RandomResizedCrop(256, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  # Preserves the number of channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply the transformation to the image
        
        img = transform(img)

        return img

    def preprocess_mask(self, mask_path):
        # Load the mask image using Pillow (PIL)
        mask = Image.open(mask_path)

        # Apply your preprocessing steps to the mask image
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Ensures the mask is a tensor
        ])
        mask = transform(mask).squeeze(dim=0).long()
        #mask = transform(mask)

        return mask

    def __getitem__(self, idx):
        # Load and preprocess image and mask
        image_name = self.ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_name}.tif")
        mask_path = os.path.join(self.mask_dir, f"{image_name} .png")


        image = self.preprocess_image(image_path)
        mask = self.preprocess_mask(mask_path)

        return {
            'image': image,
            'mask': mask
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train U-Net++ or Attention U-Net")
    parser.add_argument("--model", type=str, choices=["resunet", "attentionunet"], required=True, help="Choose the model to train")
    args = parser.parse_args()

    images_dir = 'C:\\Users\\ADMIN\\Desktop\\Pyto\\data\\kmms_training\\images'
    mask_dir = 'C:\\Users\\ADMIN\\Desktop\\Pyto\\data\\kmms_training\\masks'


    scale = 0.5  # You can adjust the scale factor as needed

    # Create an instance of CarvanaCustomDataset for training and validation
    custom_dataset = CarvanaCustomDataset(images_dir, mask_dir, scale)
    custom_dataset.shuffle_data()

    # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

    batch_size = 4 # Adjust as needed
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 5

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Validation data should not be shuffled
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.26):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, inputs, targets):
            bce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-bce_loss)
            focal_loss = (self.alpha * (1 - pt) ** self.gamma) * bce_loss
            return torch.mean(focal_loss)

    # Define Dice Loss
    '''class DiceLoss(nn.Module):
        def __init__(self):
            super(DiceLoss, self).__init__()

        def forward(self, inputs, targets):
            smooth = 1.0
            inputs = torch.sigmoid(inputs)
            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return 1 - dice'''

    # Define loss function and optimizer for ResUNet
    criterion_resunet = FocalLoss(gamma=2, alpha=0.26)
    #criterion_focal = FocalLoss(gamma=2, alpha=0.26)
    #criterion_dice = DiceLoss()

    #criterion_resunet = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    in_channels_resunet = 3  # Assuming RGB images
    out_channels_resunet = 3  # Binary segmentation (1 channel)
    resunet_model = ResUNet(in_channels_resunet, out_channels_resunet)
    resunet_model.apply(initialize_weights)

    #optimizer_resunet=optim.SGD(resunet_model.parameters(), lr=learning_rate, momentum=momentum),
    optimizer_resunet = optim.Adam(resunet_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler_resunet = ReduceLROnPlateau(optimizer_resunet, mode='min', factor=0.1, patience=5, verbose=True)

    

    best_loss_resunet = float('inf')
    best_model_weights_resunet = None
    start_time = time.time()
    resunet_train_losses = []
    resunet_val_losses = []
    attentionunet_train_losses = []
    attentionunet_val_losses = []

    # Training loop for ResUNet
    for epoch in range(num_epochs):
        train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
        resunet_model.train()
        
        for batch in train_loader:
            inputs = batch['image']
            targets = batch['mask']
            

            # Forward pass
            outputs = resunet_model(inputs)
            loss = criterion_resunet(outputs, targets)

            # Backpropagation and optimization
            optimizer_resunet.zero_grad()
            loss.backward()
            optimizer_resunet.step()
        
        # Validation loop for ResUNet
        resunet_model.eval()  # Set the model to evaluation mode
        val_loss_resunet = 0.0
        with torch.no_grad():  # Disable gradient tracking during validation
            for batch in val_loader:
                inputs = batch['image']
                targets = batch['mask']

                # Forward pass
                outputs = resunet_model(inputs)
                loss = criterion_resunet(outputs, targets)
                val_loss_resunet += loss.item()
        
        # Calculate average validation loss for ResUNet
        val_loss_resunet /= len(val_loader)
        scheduler_resunet.step(val_loss_resunet)
        resunet_train_losses.append(loss.item())
        resunet_val_losses.append(val_loss_resunet)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - ResUNet - Train Loss: {loss:.3f} - Val Loss: {val_loss_resunet:.3f}")
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] took {epoch_time} seconds")
        start_time = time.time()
        # Save the model's weights if it's the best so far
        if val_loss_resunet < best_loss_resunet:
            best_loss_resunet = val_loss_resunet
            best_model_weights_resunet = resunet_model.state_dict()

    # Save the best ResUNet model weights
    torch.save(best_model_weights_resunet, 'best_resunet_weights.pth')
    criterion_attentionunet = FocalLoss(gamma=2, alpha=0.26)


    # Define loss function and optimizer for AttentionUNet
    #criterion_attentionunet = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    in_channels_attentionunet = 3  # Assuming RGB images
    out_channels_attentionunet = 3
    attentionunet_model = AttentionUNet(in_channels_attentionunet, out_channels_attentionunet)
    attentionunet_model.apply(initialize_weights)
    optimizer_attentionunet = optim.Adam(attentionunet_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler_attentionunet = ReduceLROnPlateau(optimizer_attentionunet, mode='min', factor=0.5, patience=5, verbose=True)
    

    best_loss_attentionunet = float('inf')
    best_model_weights_attentionunet = None
    start_time = time.time()
    
    # Training loop for AttentionUNet
    for epoch in range(num_epochs):
        train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
        attentionunet_model.train()  # Set the model to training mode
        for batch in train_loader:
            inputs = batch['image']
            targets = batch['mask']
            #inputs = F.interpolate(inputs, size=(128, 128), mode='bilinear', align_corners=True)
            #targets = F.interpolate(targets, size=(128, 128), mode='bilinear', align_corners=True)

            # Forward pass
            outputs = attentionunet_model(inputs)
            loss = criterion_attentionunet(outputs, targets)

            # Backpropagation and optimization
            optimizer_attentionunet.zero_grad()
            loss.backward()
            optimizer_attentionunet.step()
        
        # Validation loop for AttentionUNet
        attentionunet_model.eval()  # Set the model to evaluation mode
        val_loss_attentionunet = 0.0
        with torch.no_grad():  # Disable gradient tracking during validation
            for batch in val_loader:
                inputs = batch['image']
                targets = batch['mask']

                # Forward pass
                outputs = attentionunet_model(inputs)
                loss = criterion_attentionunet(outputs, targets)
                val_loss_attentionunet += loss.item()
        
        # Calculate average validation loss for AttentionUNet
        val_loss_attentionunet /= len(val_loader)
        scheduler_attentionunet.step(val_loss_attentionunet)
        attentionunet_train_losses.append(loss.item())
        attentionunet_val_losses.append(val_loss_attentionunet)

        print(f"Epoch [{epoch+1}/{num_epochs}] - AttentionUNet - Train Loss: {loss:.3f} - Val Loss: {val_loss_attentionunet:.3f}")
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] took {epoch_time} seconds")
        start_time = time.time()
        # Save the model's weights if it's the best so far
        if val_loss_attentionunet < best_loss_attentionunet:
            best_loss_attentionunet = val_loss_attentionunet
            best_model_weights_attentionunet = attentionunet_model.state_dict()
        

    # Save the best AttentionUNet model weights
    torch.save(best_model_weights_attentionunet, 'best_attentionunet_weights.pth')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), resunet_train_losses, label='ResUNet Loss')
    #plt.plot(range(1, num_epochs + 1), resunet_val_losses, label='ResUNet Validation Loss')
    plt.title('ResUNet Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), attentionunet_train_losses, label='AttentionUNet Loss')
    #plt.plot(range(1, num_epochs + 1), attentionunet_val_losses, label='AttentionUNet Validation Loss')
    plt.title('AttentionUNet Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

