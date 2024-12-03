# Copyright (c) 2024, Daksitha Withanage don

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from sklearn.decomposition import PCA
from pathlib import Path
import hubconf
# Assuming 'dinov2_vitx14' is your pre-loaded model and it's already set to the correct device.

def apply_pca_across_frames(raw_features, n_components=3):
    number_of_frames, num_patches, _ = raw_features.shape
    pca_transformed_features = np.zeros((number_of_frames, num_patches, n_components))
    for patch_idx in range(num_patches):
        patch_features_across_frames = raw_features[:, patch_idx, :]
        pca = PCA(n_components=n_components)
        pca_transformed_features[:, patch_idx, :] = pca.fit_transform(patch_features_across_frames)
    return pca_transformed_features

def load_and_transform_images(image_paths, transform):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    images_tensor = torch.stack([transform(img) for img in images])
    return images_tensor

# Define your transformations
transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# List of image paths
image_dir = Path('D:/GitHub/dino/data/output/NP001/infant/frames/')
image_paths = [image_dir / f'frame_000{i}.jpg' for i in range(10)]

dinov2_vitx14 = hubconf.dinov2_vitl14()


# Move the model to CUDA device
dinov2_vitx14 = dinov2_vitx14.to('cuda')

# Load and transform images
images_tensor = load_and_transform_images(image_paths, transforms)

# Extract features in batch
with torch.no_grad():
    inference = dinov2_vitx14.forward_features(images_tensor.to('cuda'))
    features = inference['x_norm_patchtokens'].detach().cpu().numpy()  # Assuming the model outputs in this structure

# Apply PCA across frames (images)
pca_features = apply_pca_across_frames(features, n_components=3)

# Visualization
def visualize_pca_components(pca_features):
    num_images, num_patches, _ = pca_features.shape
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    for img_idx in range(num_images):
        for comp_idx in range(3):
            component_image = pca_features[img_idx, :, comp_idx].reshape(16, 16)
            ax = axes[img_idx, comp_idx] if num_images > 1 else axes[comp_idx]
            ax.imshow(component_image, cmap='inferno')
            ax.set_title(f'Image {img_idx+1} - PCA Component {comp_idx+1}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_pca_components(pca_features)
