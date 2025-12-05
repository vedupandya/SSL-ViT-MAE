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

# --- DDP SETUP/CLEANUP (Multi-Node Compatible) ---
def setup(local_rank):
    """Initializes distributed environment using Slurm variables."""
    
    if 'SLURM_NTASKS' not in os.environ or int(os.environ['SLURM_NTASKS']) <= 1:
        # Should not be called in single-GPU mode now, but included for robustness
        raise RuntimeError("Setup called in invalid DDP context.")

    world_size = int(os.environ['SLURM_NTASKS'])
    gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
    node_rank = int(os.environ['SLURM_NODEID'])
    
    # Retrieve the master hostname
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']]).decode().split()
    master_addr = hostnames[0]
    
    global_rank = node_rank * gpus_per_node + local_rank
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '12355' 
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
    
    if global_rank == 0:
        print(f"[DDP Init] World Size: {world_size}, Master: {master_addr}")

    return global_rank, world_size, gpus_per_node

def cleanup():
    """Tear down the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

# --- TRAINING WORKER FUNCTION ---
def train_worker(global_rank, world_size, local_rank, gpus_per_node, args):
    """
    Main training function run by each GPU process.
    """
    
    # Check if running in DDP mode or Single-GPU mode
    is_ddp = dist.is_initialized()
    
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
    
    # --------------------------------------------------------------------------
    # NOTE: DDP wrapper MUST happen AFTER loading weights, BUT before the training loop.
    # We will instantiate the DDP model only after the weights are loaded into raw_model.
    
    # Placeholder for DDP model until weights are loaded
    model = raw_model
    # --------------------------------------------------------------------------


    # 3. Data Loading
    transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = FlatImageDataset(TRAIN_DIR, transform)
    
    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        shuffle = False
        batch_size = BATCH_SIZE // world_size
    else:
        sampler = None
        shuffle = True
        batch_size = BATCH_SIZE

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        prefetch_factor=2,
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device='cuda')

    # 4. Resume Training (Final Optimized DDP Resume Logic)
    start_epoch = 0
    latest = None
    is_master = (global_rank == 0)

    # Dictionary to hold states loaded from disk (only necessary for Rank 0)
    map_location = {'cuda:0': f'cuda:{local_rank}'}
    
    # Find checkpoint path and broadcast existence
    if is_master:
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(ckpt_dir_path, exist_ok=True)
        latest = latest_checkpoint(ckpt_dir_path)
        
    latest_exists = torch.tensor(1, device=local_rank) if latest else torch.tensor(0, device=local_rank)
    
    if is_ddp:
        dist.broadcast(latest_exists, src=0)
    
    if latest_exists.item() == 1:
        # 4a. --- I/O Phase: Only Rank 0 loads the full state ---
        if is_master:
            # Load full checkpoint dictionary
            ckpt = torch.load(str(latest), map_location=map_location)
            
            # --- Load States (Model, Opt, Scaler) ---
            # NOTE: We load into the non-DDP raw_model and optimizer on Rank 0
            raw_model.load_state_dict(ckpt['model'], strict=True)
            optimizer.load_state_dict(ckpt['opt'])
            scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            
            # Manually ensure the LR is correct after loading the optimizer state
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
            
            print(f'Resuming from {latest} on global rank {global_rank}. Starting at Epoch {start_epoch}')
            
        # 4b. --- Synchronization Phase: Broadcast Model Weights ---
        # CRITICAL FIX: The entire raw_model state dict is broadcast from Rank 0 
        # to all other ranks to ensure uniform weight initialization.
        if is_ddp:
            dist.barrier() # Ensure all ranks are ready before syncing weights
            dist.broadcast(raw_model.state_dict(), src=0)
            
    # 5. --- Final DDP Initialization ---
    if is_ddp:
        # Wrap the model AFTER weights have been loaded and synchronized
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model # Use the simple model wrapper
    # 5. Training Loop
    if is_master and LOGGING:
        writer = SummaryWriter(LOG_DIR)
        
    eval_train_loader, eval_test_loader = None, None
    if is_master:
        eval_train_loader, eval_test_loader = build_eval_dataloaders(IMG_SIZE)


    for epoch in range(start_epoch, EPOCHS):
        if is_ddp:
            sampler.set_epoch(epoch) 

        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(loader):
            imgs = batch[0].to(local_rank)
            optimizer.zero_grad()

            pred, mask = raw_model(imgs) 
            
            per_patch = mae_loss(pred, imgs, cfg['PATCH_SIZE'])
            loss = (per_patch * mask.float()).sum() / mask.float().sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Logging only happens on master process
            if is_master and LOGGING:
                global_step = epoch * len(loader) + step
                writer.add_scalar('step/loss', loss.item(), global_step)

                if global_step > 0 and global_step % 1000 == 0:
                    model.eval()
                    with torch.no_grad():
                        x = imgs[:4].cpu() 
                        pred_raw, mask_raw = raw_model(imgs[:4].to(local_rank))
                        
                        rec = apply_mae_reconstruction(imgs[:4].to(local_rank), pred_raw, mask_raw, patch_size=cfg['PATCH_SIZE']).cpu()
                        full_pred = unpatchify(pred_raw, patch_size=cfg['PATCH_SIZE']).cpu()

                        grid = torch.cat([x, rec, full_pred], dim=0)
                        writer.add_images(f"Reconstruction/step_{global_step}", grid, global_step)
                    model.train()

        # Loss Aggregation
        if is_ddp:
            total_loss_tensor = torch.tensor([total_loss], device=local_rank)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg = (total_loss_tensor.item() / world_size) / len(loader)
        else:
            avg = total_loss / len(loader)


        if is_master:
            print(f'Epoch {epoch} avg loss {avg:.5f}')
            if LOGGING:
                writer.add_scalar('epoch/loss', avg, epoch)

            save_checkpoint(os.path.join(ckpt_dir_path, f'mae_checkpoint_{epoch}.pth'), epoch, raw_model, optimizer, scaler)
            if (epoch + 1) % 5 == 0:    
                
                # Evaluation
                eval_acc = eval_linear_probe(raw_model, eval_train_loader, eval_test_loader, local_rank, lin_epochs=LP_EPOCHS)
                if LOGGING:
                    writer.add_scalar('eval/lp_acc', eval_acc, epoch)
            
        if is_ddp:
            dist.barrier() # Ensure all processes sync before starting next epoch


    if is_master and LOGGING:
        writer.close()
        
    if is_ddp:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description='MAE Pretraining')
    parser.add_argument('--model_size', type=str, default=DEFAULT_MODEL_SIZE, 
                        choices=list(MODEL_CONFIGS.keys()), help='Model configuration size')
    args = parser.parse_args()
    
    # 1. Check Execution Environment
    if 'SLURM_NTASKS' in os.environ and int(os.environ['SLURM_NTASKS']) > 1:
        # --- Multi-GPU / Multi-Node DDP (SLURM) ---
        
        # Slurm sets SLURM_LOCALID (our local_rank) and we need to call setup
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        global_rank, world_size, gpus_per_node = setup(local_rank)
        
        train_worker(global_rank, world_size, local_rank, gpus_per_node, args)
        
    else:
        # --- Single-GPU / CPU Execution (No DDP) ---
        
        world_size = 1 
        local_rank = 0 
        global_rank = 0
        gpus_per_node = 1
        
        if torch.cuda.is_available():
            print("Starting single-GPU training (No DDP initialization).")
            torch.cuda.set_device(local_rank) 
            train_worker(global_rank, world_size, local_rank, gpus_per_node, args)
        else:
            print("Starting single-CPU training (No GPUs available).")
            # If no GPU, call the worker on CPU device 0
            train_worker(global_rank, world_size, 0, gpus_per_node, args)


if __name__ == '__main__':
    main()