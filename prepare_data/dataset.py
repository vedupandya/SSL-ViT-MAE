from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None, exts=None):
        self.root = Path(root)
        self.transform = transform
        if exts is None:
            exts = ('.jpg','.jpeg','.png','.bmp','.webp')
        self.files = sorted([p for p in self.root.iterdir() if p.suffix.lower() in exts])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0
