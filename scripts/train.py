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
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") 
    # Find checkpoint path and broadcast existence
    if is_master:
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(ckpt_dir_path, exist_ok=True)
        latest = latest_checkpoint(ckpt_dir_path)
        
    # Broadcast whether latest exists (0/1)
    flag = torch.tensor([1 if latest else 0], dtype=torch.int64, device=device)
    if is_ddp:
        dist.broadcast(flag, src=0)
    has_ckpt = bool(flag.item())

    if has_ckpt:
        # --- Rank 0: load checkpoint to CPU (robust across machines) ---
        if is_master:
            start_epoch = load_checkpoint(latest, raw_model, optimizer, scaler, map_location='cpu') + 1

            # move model parameters to device (rank0)
            raw_model.to(device)
            # ensure optimizer param_groups lr matches computed LR
            for pg in optimizer.param_groups:
                pg['lr'] = LR

        # --- All ranks: synchronize start_epoch ---
        epoch_t = torch.tensor([start_epoch], dtype=torch.int64, device=device)
        if is_ddp:
            dist.broadcast(epoch_t, src=0)
        start_epoch = int(epoch_t.item())

        # --- Broadcast model parameters (tensor-by-tensor) ---
        if is_ddp:
            dist.barrier()
            for param in raw_model.state_dict().values():
                # param is a tensor on CPU for master; ensure on device for broadcast
                # we must move param to device before broadcasting on non-master ranks
                if not param.is_cuda:
                    param.data = param.data.to(device)
                dist.broadcast(param, src=0)
            dist.barrier()

            # --- Broadcast optimizer state tensors (if any) ---
            # optimizer.state is mapping param->state; each state value is dict of tensors and scalars
            opt_state = optimizer.state_dict()
            # iterate over states and broadcast tensor entries
            for state in opt_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        # ensure tensor on device then broadcast in-place
                        if not v.is_cuda:
                            v = v.to(device)
                        dist.broadcast(v, src=0)

            # After we modified the tensor objects above, load back into optimizer
            # (re-apply the possibly-updated opt_state)
            # Note: PyTorch optimizer.state_dict() returns new objects; safer approach:
            # just call optimizer.load_state_dict(opt_state) on all ranks; keys align.
            optimizer.load_state_dict(opt_state)
            dist.barrier()
        else:
            # single gpu: ensure model on device
            raw_model.to(device)


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
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Global batch size across all GPUs')
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