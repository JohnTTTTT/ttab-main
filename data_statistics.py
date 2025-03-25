import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # progress bar

# --- Dataset Directory Setup ---
# Option 1: Use only the training set.
train_dir = '/home/johnt/scratch/AffectNet/extracted_files/train_set/separated_images'
# Option 2: If you want to compute statistics on both training and test data,
#         you could organize your dataset directory such that it contains both subdirectories.
# For example, if you have a parent directory with 'train' and 'test' folders:
# data_dir = '/home/johnt/scratch/AffectNet/extracted_files/combined_dataset'
#
# IMPORTANT: Typically, normalization parameters (mean & std) are computed only on the training set.
#            Using only the training set avoids data leakage and is standard practice.
data_dir = train_dir  # Change this to your combined directory if needed

# --- Transformation Setup ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),
])

# --- Dataset and DataLoader ---
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# --- Variables to Compute Mean and Std ---
mean = 0.0
std = 0.0
nb_samples = 0

print(f"Starting processing of {len(dataset)} images...")

# --- Processing with Progress Feedback ---
for batch_idx, (data, _) in enumerate(tqdm(loader, desc="Processing Batches"), 1):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # flatten height and width dimensions
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
    # Optional: print progress per batch (comment out if tqdm is sufficient)
    print(f"Processed batch {batch_idx}/{len(loader)}")

# --- Final Computation ---
mean /= nb_samples
std /= nb_samples

print("Computed Mean:", mean)
print("Computed Std:", std)
