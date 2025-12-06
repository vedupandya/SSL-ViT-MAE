"""
SUN397 Dataset Preparation for Kaggle Competition (Final Evaluation)
=====================================================================

Creates Kaggle competition from SUN397 (Scene Understanding):
- All 397 scene classes
- ~108k images
- Resize to 96x96 during processing (memory efficient)

Usage:
    python prepare_sun397_for_kaggle.py \
        --output_dir ./kaggle_data_sun397 \
        --resolution 96

CRITICAL FIXES:
1. Memory Safety: Only loads image metadata for splitting. Full images are
   loaded one-by-one, resized, and saved to prevent the Out-of-Memory (OOM)
   crash ("zsh: killed").
2. Stability: Forces 'spawn' multiprocessing start method to prevent 
   "leaked semaphore objects" warnings and subsequent process instability.
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing as mp

# --- CRITICAL STABILITY FIX ---
# Force the 'spawn' start method to resolve potential semaphore resource leaks
# and stability issues often encountered in data loading pipelines, which 
# were being triggered by the abrupt OOM termination.
try:
    # Use 'spawn' for better process stability and resource cleanup
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Handles the case where the start method is already set (e.g., in IPython or a notebook)
    pass
# ------------------------------

def download_sun397(download_dir):
    """
    Download SUN397 dataset using HuggingFace datasets
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        # Instruction for the user if the dependency is missing
        raise ImportError("Please install datasets: pip install datasets")
    
    print("="*60)
    print("Downloading SUN397 from HuggingFace...")
    print("This will download ~37 GB (original resolution) to the cache directory.")
    print("We will process and resize to the target resolution during saving.")
    print("="*60)
    
    # Load SUN397 from HuggingFace
    print("\nLoading dataset (this may take a while)...")
    dataset = load_dataset("tanganke/sun397", cache_dir=str(download_dir))
    
    print(f"\nDataset loaded!")
    print(f"  Train split size: {len(dataset['train'])} images")
    if 'test' in dataset:
        print(f"  Test split size: {len(dataset['test'])} images")
    
    return dataset


def load_sun397_structure(dataset):
    """
    Load SUN397 dataset structure from HuggingFace dataset
    
    HuggingFace format: {'image': PIL.Image, 'label': int}
    """
    print("\n" + "="*60)
    print("Processing SUN397 structure...")
    print("="*60)
    
    # We primarily use the 'train' split for the competition data
    train_data = dataset['train']
    
    # Get label names from the features object
    label_names = train_data.features['label'].names
    
    print(f"\nTotal classes: {len(label_names)}")
    print(f"Total images in 'train' split: {len(train_data)}")
    
    # Count images per class
    class_counts = {}
    for item in tqdm(train_data, desc="Counting classes"):
        label = item['label']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Create structured class information list
    class_info = [
        {
            'class_id': idx,
            'class_name': name,
            'n_images': class_counts.get(idx, 0)
        }
        for idx, name in enumerate(label_names)
    ]
    
    return class_info, train_data


def create_kaggle_dataset(
    hf_dataset,
    output_dir,
    class_info,
    resolution=96,
    seed=42
):
    """
    Create Kaggle competition dataset from SUN397 HuggingFace dataset.
    This function is now memory-safe by only loading images during the save step.
    
    Args:
        hf_dataset: HuggingFace dataset (train split, containing the images).
        output_dir: Output directory path.
        class_info: List of class information dicts.
        resolution: Target image resolution.
        seed: Random seed for splitting.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    print("\n" + "="*60)
    print(f"Creating SUN397 Kaggle dataset")
    print(f"  Total Images to Process: {len(hf_dataset)}")
    print(f"  Target Resolution: {resolution}x{resolution}")
    print("="*60)
    
    # --- MEMORY SAFE STEP 1: Collect only metadata (indices) ---
    print("\nCollecting metadata for splitting (Memory-safe)...")
    all_metadata = []
    
    for idx in tqdm(range(len(hf_dataset)), desc="Loading metadata"):
        item = hf_dataset[idx] # Accessing this fetches metadata/labels, not the image binary
        all_metadata.append({
            'hf_index': idx,  # CRITICAL: This index points back to the image in the HF dataset
            'class_id': item['label'],
            'class_name': class_info[item['label']]['class_name']
        })
    
    print(f"Total metadata collected: {len(all_metadata)}")
    
    # 2. Shuffle and split the METADATA
    np.random.shuffle(all_metadata)
    
    n_total = len(all_metadata)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_metadata = all_metadata[:n_train]
    val_metadata = all_metadata[n_train:n_train+n_val]
    test_metadata = all_metadata[n_train+n_val:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_metadata)} images")
    print(f"  Val: {len(val_metadata)} images")
    print(f"  Test: {len(test_metadata)} images")
    
    # 3. Save function that loads and processes images one-by-one
    def save_split(split_metadata, split_name, hf_dataset, res):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        print(f"\nSaving {split_name} images (resize to {res}x{res})...")
        for idx, item_meta in enumerate(tqdm(split_metadata, desc=f"{split_name}")):
            try:
                # Load image from HuggingFace dataset using the index
                hf_index = item_meta['hf_index']
                
                # *** IMAGE IS LOADED INTO MEMORY HERE AND IMMEDIATELY PROCESSED ***
                # This ensures only one image is fully loaded at any given time.
                img = hf_dataset[hf_index]['image'].convert('RGB')
                
                # Resize to target resolution
                # Image.Resampling.BILINEAR is more descriptive in modern PIL
                img = img.resize((res, res), Image.Resampling.BILINEAR)
                
                # Save with new filename format
                filename = f"{idx:06d}_class{item_meta['class_id']:03d}.jpg"
                img.save(split_dir / filename, quality=85)
                
                metadata.append({
                    'filename': filename,
                    'class_id': item_meta['class_id'],
                    'class_name': item_meta['class_name']
                })
                
                # Explicitly delete the image object to free RAM immediately
                del img
                
            except Exception as e:
                print(f"\nWarning: Failed to process image (HF index {hf_index}): {e}")
                continue
        
        return pd.DataFrame(metadata)
    
    # Save all splits, passing the metadata list and the full hf_dataset object
    train_df = save_split(train_metadata, 'train', hf_dataset, resolution)
    val_df = save_split(val_metadata, 'val', hf_dataset, resolution)
    test_df = save_split(test_metadata, 'test', hf_dataset, resolution)
    
    # Save CSVs
    train_df.to_csv(output_dir / 'train_labels.csv', index=False)
    val_df.to_csv(output_dir / 'val_labels.csv', index=False)
    
    # Test: save internal labels and public version
    test_df.to_csv(output_dir / 'test_labels_INTERNAL.csv', index=False)
    test_df[['filename']].to_csv(output_dir / 'test_images.csv', index=False)
    
    # Create sample submission
    create_sample_submission(test_df, output_dir)
    
    # Create class mapping file
    class_mapping_df = pd.DataFrame(class_info)
    class_mapping_df.to_csv(output_dir / 'class_mapping.csv', index=False)
    
    # Create README
    create_readme(output_dir, len(class_info), resolution)
    
    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    return train_df, val_df, test_df


def create_sample_submission(test_df, output_dir):
    """Create sample_submission.csv"""
    output_dir = Path(output_dir)
    
    sample_submission = pd.DataFrame({
        'id': test_df['filename'],
        'class_id': 0 # Default class ID for sample submission
    })
    
    sample_submission.to_csv(output_dir / 'sample_submission.csv', index=False)
    print(f"\nSample submission created")


def create_readme(output_dir, num_classes, resolution):
    """Create README for dataset"""
    readme = f"""# SUN397 Scene Recognition Dataset (Final Evaluation)

## Dataset Info

- **Source**: SUN397 (Scene UNderstanding)
- **Classes**: {num_classes} scene categories
- **Resolution**: {resolution}x{resolution}
- **Task**: Scene recognition (indoor/outdoor environments)

## Files

- `train/` - Training images with labels
- `val/` - Validation images with labels
- `test/` - Test images (no labels)
- `train_labels.csv` - Training labels
- `val_labels.csv` - Validation labels
- `test_images.csv` - Test image list (no labels)
- `class_mapping.csv` - Class ID to scene name mapping
- `sample_submission.csv` - Example submission format

## Scene Categories

Examples: abbey, airport_terminal, bedroom, beach, forest, kitchen, mountain, 
office, restaurant, street, subway, etc.

## Submission Format

```csv
id,class_id
000001_class042.jpg,42
000002_class015.jpg,15
...
```

## Citation

J. Xiao, J. Hays, K. Ehinger, A. Oliva, and A. Torralba.
SUN Database: Large-scale Scene Recognition from Abbey to Zoo.
IEEE Conference on Computer Vision and Pattern Recognition, 2010.
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='Prepare SUN397 for Kaggle (Final Eval)')
    parser.add_argument('--download_dir', type=str, default='./raw_data',
                        help='Directory to cache HuggingFace dataset')
    parser.add_argument('--output_dir', type=str, default='./kaggle_data_sun397',
                        help='Output directory for Kaggle dataset')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Target image resolution (96, 224, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SUN397 Kaggle Competition Preparation (Final Evaluation)")
    print("="*60)
    print("\nDependencies check: Requires 'pandas', 'numpy', 'Pillow', 'tqdm', 'datasets'")
    print("If you haven't already: pip install pandas numpy pillow tqdm datasets")
    print()
    
    # Download SUN397 from HuggingFace
    dataset = download_sun397(args.download_dir)
    
    # Load dataset structure
    class_info, train_data = load_sun397_structure(dataset)
    
    # Create Kaggle dataset (Now memory-safe!)
    train_df, val_df, test_df = create_kaggle_dataset(
        train_data,
        args.output_dir,
        class_info,
        resolution=args.resolution,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()