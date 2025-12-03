import torch
from pathlib import Path
import re # Added import

def latest_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    # --- FIX: sort by epoch number (prevents issues with epoch 10 vs epoch 2) ---
    files = ckpt_dir.glob('mae_checkpoint_*.pth')
    
    # Extract epoch number from filename (e.g., 'mae_checkpoint_123.pth' -> 123)
    epoch_files = []
    for f in files:
        match = re.search(r'mae_checkpoint_(\d+)\.pth', f.name)
        if match:
            epoch = int(match.group(1))
            epoch_files.append((epoch, f))
            
    if not epoch_files:
        return None
        
    # Sort by epoch number (descending) and return the path of the latest
    latest_file = sorted(epoch_files, key=lambda x: x[0], reverse=True)[0]
    return latest_file[1]
    # -------------------------------------------------------------------------------------

def save_checkpoint(path, epoch, model, optimizer, scaler):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'])
    if optimizer: optimizer.load_state_dict(ckpt['opt'])
    if scaler: scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0)