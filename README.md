# JoinUNet Project

This repository contains code and instructions for executing the JoinUNet project.

## Project Structure

- `JoinUNet/`
  - `kmms_training/`
    - `images/` - Training images.
    - `masks/` - Training masks.
  - `output/` - Predicted images.
  - `test/`
    - `images/` - Test images.
    - `masks/` - Test masks.

## Files

- `ensemble_inference.py` - Code for ensemble inference.
- `train.py` - Code for training the model.
- `predict.py` - Code for making predictions.
- `unet.py` - Code for the UNet model.
- `best_resunet_weights.pth` - Pre-trained weights for the ResUNet model.
- `best_attentionunet_weights.pth` - Pre-trained weights for the Attention UNet model.
- `ensemble.pth` - Pre-trained ensemble model.

## Execution

To execute the code, follow these steps:
## Command Prompt Execution

To run the code in the JoinUNet project using a command prompt, follow these steps:

### 1. Navigate to the Project Folder

Open your command prompt and navigate to the JoinUNet project folder using the `cd` (change directory) command:
```bash
cd path/to/JoinUNet
```

### 2. Model Training

To train both the ResUNet and Attention UNet models, you can use the following command:
```bash
python train.py --model resunet/attentionunet
```

### 3. Ensemble Inference

Once the training is complete, you can perform ensemble inference and save the ensemble.pth file:
```bash
python ensemble_inference.py
```

### 4. Generating Predictions

Now, to generate predictions on test images using the ensemble model, use the following command:
```bash
python predict.py --model path-to-the-model\ensemble.pth --test-folder path-to-testimages-folder\images --output-folder path-to-save-predicted-masks\output
```
Make sure to replace path-to-the-model, path-to-testimages-folder, and path-to-save-predicted-masks with the actual paths on your system.

By following these steps, you can execute the JoinUNet project via the command prompt, including model training, ensemble inference, and generating predictions on test images.



