import torch
from glob import glob
from pathlib import Path

def latest_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    files = sorted(ckpt_dir.glob('mae_checkpoint_*.pth'))
    return files[-1] if files else None

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
