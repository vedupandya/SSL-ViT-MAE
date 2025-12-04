import os
import sys
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import LOG_DIR, CKPT_DIR, TRAIN_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE, EPOCHS, LOGGING, MODEL_CONFIGS, DEFAULT_MODEL_SIZE, LP_EPOCHS

from prepare_data.dataset import FlatImageDataset
import torchvision.transforms as T
from models.mae import MAE, mae_loss
from utils.vision import unpatchify, apply_mae_reconstruction
from utils.checkpoint import latest_checkpoint, save_checkpoint, load_checkpoint
from eval import build_eval_dataloaders, eval_linear_probe, eval_knn

torch.manual_seed(42)

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Use a random unused port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Tear down the distributed environment."""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """
    Main training function run by each GPU process.
    rank: The current GPU's ID (0 or 1, etc.)
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 1. Configuration Setup
    cfg = MODEL_CONFIGS[args.model_size]
    LR = cfg['LR']*BATCH_SIZE/256  # Scale LR based on global batch size
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    
    # Checkpoint directory for this model size
    global CKPT_DIR
    ckpt_dir_path = CKPT_DIR / args.model_size
    
    # 2. Model Initialization
    raw_model = MAE(
        img_size=IMG_SIZE,
        patch_size=cfg['PATCH_SIZE'],
        enc_dim=cfg['ENC_DIM'],
        enc_depth=cfg['ENC_DEPTH'],
        enc_heads=cfg['ENC_HEADS'],
        dec_dim=cfg['DEC_DIM'],
        dec_depth=cfg['DEC_DEPTH'],
        dec_heads=cfg['DEC_HEADS'],
        mask_ratio=cfg['MASK_RATIO'],
    ).to(rank)
    
    # Wrap the model with DDP
    model = DDP(raw_model, device_ids=[rank])

    # 3. Data Loading
    transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = FlatImageDataset(TRAIN_DIR, transform)
    
    # CRITICAL DDP CHANGE: Use DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE // world_size, # Divide global batch size by number of GPUs
        shuffle=False, # Shuffle handled by sampler
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        prefetch_factor=2,
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device='cuda')

    # 4. Resume Training
    start_epoch = 0
    if rank == 0:
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(ckpt_dir_path, exist_ok=True)
        latest = latest_checkpoint(ckpt_dir_path)
        if latest:
            start_epoch = load_checkpoint(str(latest), raw_model, optimizer, scaler, map_location=f'cuda:{rank}') + 1
            print(f'Resuming from {latest} on rank {rank}')

    dist.barrier() # Ensure all processes sync after loading checkpoints

    # 5. Training Loop
    if rank == 0 and LOGGING:
        writer = SummaryWriter(LOG_DIR)
        
    eval_train_loader, eval_test_loader = None, None
    if rank == 0:
        eval_train_loader, eval_test_loader = build_eval_dataloaders(IMG_SIZE)


    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch) # Critical for DDP shuffling
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(loader):
            imgs = batch[0].to(rank)
            optimizer.zero_grad()

            pred, mask = raw_model(imgs) # Use raw_model inside the forward pass
            
            per_patch = mae_loss(pred, imgs, cfg['PATCH_SIZE'])
            loss = (per_patch * mask.float()).sum() / mask.float().sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Logging and checkpointing only happens on rank 0
            if rank == 0 and LOGGING:
                global_step = epoch * len(loader) + step
                writer.add_scalar('step/loss', loss.item(), global_step)

                # Simplified image logging check for DDP compatibility
                if global_step > 0 and global_step % 5000 == 0:
                    model.eval()
                    with torch.no_grad():
                        x = imgs[:4].cpu() # Use a few images from the current GPU
                        pred_raw, mask_raw = raw_model(imgs[:4].to(rank))
                        
                        rec = apply_mae_reconstruction(imgs[:4].to(rank), pred_raw, mask_raw, patch_size=cfg['PATCH_SIZE']).cpu()
                        full_pred = unpatchify(pred_raw, patch_size=cfg['PATCH_SIZE']).cpu()

                        grid = torch.cat([x, rec, full_pred], dim=0)
                        writer.add_images(f"Reconstruction/step_{global_step}", grid, global_step)
                    model.train()

        # Gather loss from all ranks (optional, but good for accurate logging)
        dist.reduce(torch.tensor([total_loss], device=rank), dst=0, op=dist.ReduceOp.SUM)
        avg = total_loss / len(loader) if world_size == 1 else (total_loss.item() / world_size) / len(loader)

        if rank == 0:
            print(f'Epoch {epoch} avg loss {avg:.5f}')
            if LOGGING:
                writer.add_scalar('epoch/loss', avg, epoch)

            if (epoch + 1) % 5 == 0:    
                save_checkpoint(os.path.join(ckpt_dir_path, f'mae_checkpoint_{epoch}.pth'), epoch, raw_model, optimizer, scaler)
                
                # Evaluation
                eval_acc = eval_linear_probe(raw_model, eval_train_loader, eval_test_loader, rank, lin_epochs=LP_EPOCHS)
                if LOGGING:
                    writer.add_scalar('eval/lp_acc', eval_acc, epoch)

    if rank == 0 and LOGGING:
        writer.close()
        
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='MAE Pretraining')
    parser.add_argument('--model_size', type=str, default=DEFAULT_MODEL_SIZE, 
                        choices=list(MODEL_CONFIGS.keys()), help='Model configuration size')
    args = parser.parse_args()
    
    # Determine world size based on environment variables if running via srun/sbatch
    world_size = int(os.environ.get('SLURM_GPUS_ON_NODE', 1))

    # If running with multiple GPUs, start the multi-process DDP wrapper
    if world_size > 1:
        print(f"Starting DDP training with {world_size} GPUs...")
        mp.spawn(train_worker, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        print("Starting single-GPU training...")
        train_worker(0, 1, args)

if __name__ == '__main__':
    main()