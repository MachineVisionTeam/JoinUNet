import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from unet import ResUNet, AttentionUNet
import numpy as np
import cv2
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from ensemble_inference import EnsembleModel
from sklearn.preprocessing import label_binarize

# Define a function to preprocess input images
def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img
def predict_masks(input_image, ensemble_model):
    input_image = preprocess_image(input_image)

    with torch.no_grad():
        ensemble_output = ensemble_model(input_image.unsqueeze(0))
        ensemble_output = torch.sigmoid(ensemble_output)  # Apply sigmoid to get probability scores

    return ensemble_output
def calculate_metrics(ground_truth_mask_array, predicted_mask_bin):
    iou = jaccard_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted')
    dice_coefficient = f1_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted')
    precision = precision_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted', zero_division=1.0)
    recall = recall_score(ground_truth_mask_array.flatten(), predicted_mask_bin.flatten(), average='weighted', zero_division=1.0)

    return iou, dice_coefficient, precision, recall
def load_ground_truth_masks_for_class(class_index, test_image_filename):
    test_image_filename_no_space = test_image_filename.replace(" ", "")
    # Construct the full path to the ground truth mask for the specific class
    ground_truth_filename = os.path.splitext(test_image_filename_no_space)[0] + ".png"
    ground_truth_path = os.path.join(r"C:\Users\ADMIN\Desktop\Pyto\data\kmms_test\kmms_test\masks", ground_truth_filename)

    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth mask '{ground_truth_path}' not found for class {class_index}.")

    ground_truth_mask = Image.open(ground_truth_path)

    # Convert the ground truth mask to a binary array (0s and 1s)
    ground_truth_mask_array = np.array(ground_truth_mask) > 0

    return ground_truth_mask_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict segmentation masks using an ensemble model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the ensemble model (e.g., 'ensemble.pth').")
    parser.add_argument("--test-folder", type=str, required=True, help="Path to the folder containing test images.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder where you want to save the test images.")
    
    args = parser.parse_args()

    # Load the ResUNet and AttentionUNet models
    resunet_model = ResUNet(in_channels=3, out_channels=3)
    attentionunet_model = AttentionUNet(in_channels=3, out_channels=3)

    # Create an instance of the EnsembleModel and load the state dict
    ensemble_model = EnsembleModel(resunet_model, attentionunet_model)
    ensemble_model.load_state_dict(torch.load(args.model))
    ensemble_model.eval()  # Set the model to evaluation mode

    # Get a list of test image file names
    test_image_folder = args.test_folder
    test_image_filenames = os.listdir(test_image_folder)
    iou_scores = []
    dice_coefficient_scores = []
    precision_scores = []
    recall_scores = []

    # Initialize lists to store ROC AUC scores for each class
    roc_auc_scores = []

    for test_image_filename in test_image_filenames:
        # Construct the full path to the test image
        test_image_path = os.path.join(test_image_folder, test_image_filename)
        test_image_filename_no_space = test_image_filename.replace(" ", "")
        predicted_mask = predict_masks(test_image_path, ensemble_model)
        # Predict the mask for the test image
        input_image = preprocess_image(test_image_path)
        with torch.no_grad():
            ensemble_output = ensemble_model(input_image.unsqueeze(0))
            ensemble_output = torch.sigmoid(ensemble_output)  # Apply sigmoid to get probability scores

        predicted_probabilities = ensemble_output.squeeze(0).cpu().numpy()
        predicted_probabilities = predicted_probabilities.reshape(-1, 1).astype(np.uint8)


        # Load the ground truth mask for the test image
        ground_truth_filename = os.path.splitext(test_image_filename_no_space)[0] + ".png"
        ground_truth_path = os.path.join(r"C:\Users\ADMIN\Desktop\Pyto\data\kmms_test\kmms_test\masks", ground_truth_filename)

        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground truth mask '{ground_truth_path}' not found.")

        ground_truth_mask = Image.open(ground_truth_path)
        predicted_mask_image = transforms.ToPILImage()(predicted_mask.squeeze(0))
        predicted_mask_image = predicted_mask_image.resize(ground_truth_mask.size).convert(ground_truth_mask.mode)

        # Overlay the predicted mask on the input image
        overlay_image = Image.blend(
            ImageOps.grayscale(predicted_mask_image),
            ImageOps.grayscale(ground_truth_mask),
            alpha=0.7
        )

        # Save the overlay image to the output folder
        output_image_path = os.path.join(args.output_folder, f"{os.path.splitext(test_image_filename)[0]}_predicted_mask.png")
        overlay_image.save(output_image_path)
        print(f"Predicted mask overlay saved to '{output_image_path}'")

        # Calculate evaluation metrics for the test image
        overlay_image_array = np.array(overlay_image)
        ground_truth_mask_array = np.array(ground_truth_mask)

        # Apply adaptive thresholding
        adaptive_thresholded = cv2.adaptiveThreshold(overlay_image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)

        # Resize the adaptive thresholded mask to match the ground truth dimensions
        try:
            adaptive_thresholded = cv2.resize(adaptive_thresholded, (ground_truth_mask_array.shape[1], ground_truth_mask_array.shape[0]))
        except cv2.error:
            print("Error during resizing. Using the original size.")

        # Ensure data types match
        threshold_value = 0.5  # Adjust the threshold value if needed
        adaptive_thresholded = (adaptive_thresholded > threshold_value).astype(int)
        num_classes = predicted_probabilities.shape[1]

        # Convert the ground truth mask to a binary array (0s and 1s)
        ground_truth_mask_array = np.array(ground_truth_mask) > 0
        ground_truth_masks = []  # Initialize an empty list
        # Flatten the ground truth mask and predicted binary mask
        

        for class_index in range(num_classes):
            # Load the ground truth mask for the specific class
            ground_truth_mask_class = load_ground_truth_masks_for_class(class_index, test_image_filename)
            ground_truth_mask_class = np.array(Image.fromarray(ground_truth_mask_class).resize((256, 256)))
            threshold = 0.5  # You can adjust the threshold as needed
            
            predicted_binary = (predicted_probabilities[:, class_index] > threshold).astype(int)
            #predicted_binary = predicted_binary[:256]
            ground_truth_mask_class = ground_truth_mask_class[:192, :256]  # Adjust the dimensions as needed
            
            ground_truth_flat = ground_truth_mask_class.ravel()
            predicted_binary_flat = predicted_binary.ravel()


            roc_auc = roc_auc_score(ground_truth_flat, predicted_binary_flat)
            roc_auc_scores.append(roc_auc)
            #print(f"ROC AUC for Class {class_index+1}: {roc_auc}")
    iou, dice_coefficient, precision, recall = calculate_metrics(ground_truth_mask_array, adaptive_thresholded)
    iou_scores.append(iou)
    dice_coefficient_scores.append(dice_coefficient)
    precision_scores.append(precision)
    recall_scores.append(recall)

    # Calculate the average values of evaluation metrics
    avg_iou = np.mean(iou_scores)
    avg_dice_coefficient = np.mean(dice_coefficient_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    mean_roc_auc = np.mean(roc_auc_scores)
    print(f"Average IoU (Jaccard Score): {avg_iou}")
    print(f"Average Dice Coefficient: {avg_dice_coefficient}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average ROC AUC: {mean_roc_auc}")
    









