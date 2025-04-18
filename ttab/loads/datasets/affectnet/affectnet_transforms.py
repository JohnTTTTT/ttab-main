import torchvision.transforms as transforms

def get_transform_affectnet(input_size=224):
    """
    Returns train and validation transforms for AffectNet.
    The training transform includes standard augmentations used in recent papers.
    """
    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}
    
    # Training augmentations:
    train_transform = transforms.Compose([
        # Randomly crop and resize. This simulates zoom and aspect ratio variations.
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        # Random horizontal flip (typical for face images).
        transforms.RandomHorizontalFlip(),
        # Color jitter to vary brightness, contrast, saturation, and hue.
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # Random rotation by up to ±15 degrees.
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
    ])

    # Validation augmentations (deterministic)
    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # Resize slightly larger than crop size.
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
    ])
    
    return train_transform, val_transform
