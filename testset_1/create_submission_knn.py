"""
Create Kaggle Submission with KNN Classifier
=============================================

Usage:
    python create_submission_knn.py \
        --data_dir ./kaggle_data \
        --output submission.csv \
        --k 5
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse

import os
import sys

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.mae import MAE
from scripts.eval import extract_features

torch.manual_seed(42)
# ============================================================================
#                          MODEL SECTION (Modular)
# ============================================================================

class FeatureExtractor:
    """
    Modular feature extractor
    """
    
    def __init__(self, model_path="mae_checkpoint.pth", device='cuda'):
        """
        Initialize feature extractor.
        
        Args:
            model_path: Path to trained SSL model checkpoint
            device: 'cuda' or 'cpu'
        """
        print(f"Loading model: {model_path}")
        self.model = MAE().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device)["model"],strict=True)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        
    def extract_features(self, image):
        """
        Extract features from a single PIL Image.
        
        Args:
            image: PIL Image
        
        Returns:
            features: numpy array of shape (feature_dim,)
        """
        image = image.to(self.device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = extract_features(self.model, image, pool="mean")
        
        return features.cpu().numpy()
        
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of PIL Images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            features: numpy array of shape (batch_size, feature_dim)
        """
        if isinstance(images, list):
            imgs = []
            for img in images:
                imgs.append(img)
            images = torch.stack(imgs, dim=0)
        
        images = images.to(self.device)
        with torch.no_grad():
            features = extract_features(self.model, images, pool="mean") # this fxn is already vectorized in eval.py
        
        return features.cpu().numpy()


# ============================================================================
#                          DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution (96 for competition, 224 for DINO baseline)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
        self.transform =  T.Compose([
            T.RandomResizedCrop(96, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load and resize image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        if self.labels is not None:
            return image, int(self.labels[idx]), img_name
        return image, img_name


# def collate_fn(batch):
#     """Custom collate function to handle PIL images"""
#     if len(batch[0]) == 3:  # train/val (image, label, filename)
#         images = [item[0] for item in batch]
#         labels = [item[1] for item in batch]
#         filenames = [item[2] for item in batch]
#         return images, labels, filenames
#     else:  # test (image, filename)
#         images = [item[0] for item in batch]
#         filenames = [item[1] for item in batch]
#         return images, filenames
def collate_fn(batch):
    """
    Collate that returns:
      - train/val: (images_tensor, labels_tensor, filenames_list)
      - test: (images_tensor, filenames_list)

    This enforces that images are batched tensors (B,3,H,W).
    """
    # Determine tuple length reliably
    sample0 = batch[0]
    if len(sample0) == 3:
        imgs = []
        labels = []
        fnames = []
        for img, lab, fname in batch:
            # If dataset accidentally returned PIL, convert here (defensive)
            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)
            imgs.append(img)
            labels.append(lab)
            fnames.append(fname)
        images = torch.stack(imgs, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, fnames

    else:  # test set: (image, filename)
        imgs = []
        fnames = []
        for item in batch:
            img, fname = item
            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)
            imgs.append(img)
            fnames.append(fname)
        images = torch.stack(imgs, dim=0)
        return images, fnames


# ============================================================================
#                          FEATURE EXTRACTION
# ============================================================================

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    """
    Extract features from a dataloader.
    
    Args:
        feature_extractor: FeatureExtractor instance
        dataloader: DataLoader
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (or None for test)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in dataloader:
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels.numpy().tolist())
        else:  # test
            images, filenames = batch
        
        # Extract features for batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# ============================================================================
#                          KNN CLASSIFIER
# ============================================================================

def train_knn_classifier(train_features, train_labels, val_features, val_labels, k=5):
    """
    Train KNN classifier on features.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        k: Number of neighbors
    
    Returns:
        classifier: Trained KNN classifier
    """
    print(f"\nTraining KNN classifier (k={k})...")
    
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Weight by inverse distance
        metric='cosine',  # Cosine similarity for embeddings
        n_jobs=-1
    )
    
    classifier.fit(train_features, train_labels)
    
    # Evaluate
    train_acc = classifier.score(train_features, train_labels)
    val_acc = classifier.score(val_features, val_labels)
    
    print(f"\nKNN Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return classifier
from sklearn.linear_model import LogisticRegression

def train_linear_probe_classifier(train_features, train_labels,
                                  val_features, val_labels,
                                  C=10.0, max_iter=2000):
    """
    Train a linear probe (multinomial logistic regression) on features.

    Args:
        train_features: numpy array (N_train, D)
        train_labels: list or array (N_train,)
        val_features: numpy array (N_val, D)
        val_labels: list or array (N_val,)
        C: inverse regularization (larger = stronger fitting)
        max_iter: training iterations

    Returns:
        classifier: trained sklearn model with .predict()
    """
    print("\nTraining Linear Probe (Logistic Regression)...")

    clf = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        verbose=0
    )

    clf.fit(train_features, train_labels)

    # Evaluate
    train_acc = clf.score(train_features, train_labels)
    val_acc = clf.score(val_features, val_labels)

    print("\nLinear Probe Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")

    return clf


# ============================================================================
#                          SUBMISSION CREATION
# ============================================================================

def create_submission(test_features, test_filenames, classifier, output_path):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained KNN classifier
        output_path: Path to save submission.csv
    """
    print("\nGenerating predictions on test set...")
    predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df['class_id'].max() <= 199, "Invalid class_id > 199"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with KNN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--model_path', type=str, default='mae_checkpoint.pth',
                        help='Path to trained SSL model checkpoint')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution (96 for competition, 224 for DINO)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_linear_probe', action='store_true',
                        help='Use linear probe instead of KNN for classification')
    parser.add_argument('--lin_C', type=float, default=10.0,
                        help='Inverse regularization for linear probe')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets (resolution={args.resolution}px)...")
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        labels=None,
        resolution=args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(args.model_path, device)
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, 'test'
    )
    
    if args.use_linear_probe:
        classifier = train_linear_probe_classifier(
            train_features, train_labels,
            val_features, val_labels,
            C=args.lin_C
        )
    else:
        # Train KNN classifier
        classifier = train_knn_classifier(
            train_features, train_labels,
            val_features, val_labels,
            k=args.k
        )
    
    # Create submission
    create_submission(test_features, test_filenames, classifier, args.output)
    
    print("\n" + "="*60)
    print("DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)
    # print("\nREMINDER: This baseline uses pretrained weights!")
    # print("For the competition, you MUST train your own SSL model from scratch.")


if __name__ == "__main__":
    main()

