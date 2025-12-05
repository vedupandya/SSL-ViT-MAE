import os
import sys
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess 
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

# --- Multi-Node DDP Setup Function ---
def setup(local_rank):
    """
    Initialize the distributed environment using Slurm environment variables.
    This function sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE, and GLOBAL_RANK.
    """
    
    # 1. Check for Slurm environment variables
    if 'SLURM_NTASKS' not in os.environ:
        # Fallback for local debugging (single-GPU training)
        world_size = torch.cuda.device_count()
        if world_size == 0: world_size = 1 # CPU fallback
        
        master_addr = 'localhost'
        gpus_per_node = world_size
        node_rank = 0
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = '12355'
        global_rank = local_rank
    
    else:
        # Slurm environment detected (Multi-node or Multi-GPU)
        
        # Get total number of processes (GPUs) and number of nodes
        world_size = int(os.environ['SLURM_NTASKS'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        node_rank = int(os.environ['SLURM_NODEID'])
        
        # Retrieve the master hostname (first node in the job allocation)
        try:
            hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']]).decode().split()
            master_addr = hostnames[0]
        except Exception as e:
            # Fallback for safety, though should work on standard Slurm
            print(f"Error fetching master hostname via scontrol: {e}. Using localhost as fallback.")
            master_addr = 'localhost'
        
        # Calculate Global Rank: position within all GPUs
        global_rank = node_rank * gpus_per_node + local_rank
        
        # Set Environment Variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = '12355' # Consistent port
        os.environ['WORLD_SIZE'] = str(world_size)

    # 2. Initialize Process Group
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        
        if global_rank == 0:
            print(f"[DDP Init] World Size: {world_size}, Master: {master_addr}")
    else:
        # CPU initialization (rarely used for training)
        dist.init_process_group("gloo", rank=global_rank, world_size=world_size)

    return global_rank, world_size, gpus_per_node

def cleanup():
    """Tear down the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_worker(global_rank, world_size, local_rank, gpus_per_node, args):
    """
    Main training function run by each GPU process.
    """
    
    # 1. Configuration Setup
    cfg = MODEL_CONFIGS[args.model_size]

    global EPOCHS, BATCH_SIZE
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    LR = cfg['LR']*BATCH_SIZE/256
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
    ).to(local_rank)
    
    # Wrap the model with DDP
    model = DDP(raw_model, device_ids=[local_rank])

    # 3. Data Loading
    transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = FlatImageDataset(TRAIN_DIR, transform)
    
    # CRITICAL DDP CHANGE: Use DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE // world_size, # Divide global batch size by total GPUs
        shuffle=False, # Shuffle handled by sampler
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        prefetch_factor=2,
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device='cuda')

    # 4. Resume Training (Only Rank 0 handles checkpoint logic)
    start_epoch = 0
    if global_rank == 0:
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(ckpt_dir_path, exist_ok=True)
        latest = latest_checkpoint(ckpt_dir_path)
        if latest:
            # Note: We load into the raw_model, not the DDP-wrapped 'model'
            start_epoch = load_checkpoint(str(latest), raw_model, optimizer, scaler, map_location=f'cuda:{local_rank}') + 1
            print(f'Resuming from {latest} on global rank {global_rank}')

    dist.barrier() # Ensure all processes sync after loading checkpoints

    # 5. Training Loop (Only Rank 0 handles TensorBoard)
    if global_rank == 0 and LOGGING:
        writer = SummaryWriter(LOG_DIR)
        
    eval_train_loader, eval_test_loader = None, None
    if global_rank == 0:
        eval_train_loader, eval_test_loader = build_eval_dataloaders(IMG_SIZE)


    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch) # Critical for DDP shuffling
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(loader):
            imgs = batch[0].to(local_rank)
            optimizer.zero_grad()

            pred, mask = raw_model(imgs) # Use raw_model inside the forward pass
            
            per_patch = mae_loss(pred, imgs, cfg['PATCH_SIZE'])
            loss = (per_patch * mask.float()).sum() / mask.float().sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Logging and checkpointing only happens on rank 0
            if global_rank == 0 and LOGGING:
                global_step = epoch * len(loader) + step
                writer.add_scalar('step/loss', loss.item(), global_step)

                if global_step > 0 and global_step % 10000 == 0:
                    model.eval()
                    with torch.no_grad():
                        x = imgs[:4].cpu() # Use a few images from the current GPU
                        pred_raw, mask_raw = raw_model(imgs[:4].to(local_rank))
                        
                        rec = apply_mae_reconstruction(imgs[:4].to(local_rank), pred_raw, mask_raw, patch_size=cfg['PATCH_SIZE']).cpu()
                        full_pred = unpatchify(pred_raw, patch_size=cfg['PATCH_SIZE']).cpu()

                        grid = torch.cat([x, rec, full_pred], dim=0)
                        writer.add_images(f"Reconstruction/step_{global_step}", grid, global_step)
                    model.train()

        # Gather loss from all ranks 
        total_loss_tensor = torch.tensor([total_loss], device=local_rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg = (total_loss_tensor.item() / world_size) / len(loader)

        if global_rank == 0:
            print(f'Epoch {epoch} avg loss {avg:.5f}')
            if LOGGING:
                writer.add_scalar('epoch/loss', avg, epoch)

            save_checkpoint(os.path.join(ckpt_dir_path, f'mae_checkpoint_{epoch}.pth'), epoch, raw_model, optimizer, scaler)
                
            if (epoch + 1) % 5 == 0:    
                # Evaluation
                eval_acc = eval_linear_probe(raw_model, eval_train_loader, eval_test_loader, local_rank, lin_epochs=LP_EPOCHS)
                if LOGGING:
                    writer.add_scalar('eval/lp_acc', eval_acc, epoch)

    if global_rank == 0 and LOGGING:
        writer.close()
        
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='MAE Pretraining')
    parser.add_argument('--model_size', type=str, default=DEFAULT_MODEL_SIZE, 
                        choices=list(MODEL_CONFIGS.keys()), help='Model configuration size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Global batch size across all GPUs')
    args = parser.parse_args()
    
    # --- Multi-Node Launch Logic ---
    if 'SLURM_NTASKS' in os.environ and int(os.environ['SLURM_NTASKS']) > 0:
        # Running via srun/sbatch (HPC DDP)
        
        world_size = int(os.environ['SLURM_NTASKS'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        
        if world_size > gpus_per_node:
            print(f"Starting Multi-Node DDP training with {world_size} tasks across {os.environ['SLURM_NNODES']} nodes...")
        else:
            print(f"Starting Single-Node DDP training with {world_size} GPUs...")

        # Slurm sets SLURM_LOCALID for us (0, 1, 2, 3...)
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        global_rank, world_size, _ = setup(local_rank)
        
        train_worker(global_rank, world_size, local_rank, gpus_per_node, args)
        
    else:
        # Fallback for local debugging (non-Slurm environment, uses mp.spawn)
        world_size = torch.cuda.device_count()
        if world_size > 0:
            print(f"Starting local DDP training with {world_size} GPUs (DEBUG MODE)...")
            mp.spawn(train_worker, args=(world_size, world_size, world_size, args,), nprocs=world_size, join=True)
        else:
            print("No GPUs found. Running on CPU (not recommended for training).")
            # Run single CPU process
            train_worker(0, 1, 0, 1, args)
        cleanup()

if __name__ == '__main__':
    main()