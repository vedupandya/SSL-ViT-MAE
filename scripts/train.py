import os
import torch
from torch.utils.tensorboard import SummaryWriter
from config import LOG_DIR, CKPT_DIR, TRAIN_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE, LR, WEIGHT_DECAY, EPOCHS, PATCH_SIZE, LOGGING
from prepare_data.dataset import FlatImageDataset
import torchvision.transforms as T
from models.mae import MAE, mae_loss
from utils.vision import unpatchify, make_masked_image, apply_mae_reconstruction
from utils.checkpoint import latest_checkpoint, save_checkpoint, load_checkpoint
from eval import build_eval_dataloaders, eval_linear_probe, eval_knn


def train_main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = FlatImageDataset(TRAIN_DIR, transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    
    eval_train_loader, eval_test_loader = build_eval_dataloaders(IMG_SIZE)
    if LOGGING:
        writer = SummaryWriter(LOG_DIR)

    model = MAE(img_size=IMG_SIZE, patch_size=PATCH_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(DEVICE)

    start_epoch = 0
    latest = latest_checkpoint(CKPT_DIR)
    if latest:
        start_epoch = load_checkpoint(str(latest), model, optimizer, scaler, map_location=DEVICE) + 1
        print('Resuming from', latest)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(loader):
            imgs = batch[0].to(DEVICE)
            optimizer.zero_grad()

            pred, mask = model(imgs)
            per_patch = mae_loss(pred, imgs, PATCH_SIZE)
            loss = (per_patch * mask.float()).sum() / mask.float().sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if LOGGING:
                global_step = epoch * len(loader) + step
                writer.add_scalar('step/loss', loss.item(), global_step)

                if global_step > 0 and global_step % 1000 == 0:
                    model.eval()
                    with torch.no_grad():
                        x = imgs[:4]

                        pred, mask = model(x)  # pred: (B,N,D), mask: (B,N)

                        # 1. original image
                        orig = x.cpu()

                        # 2. masked image
                        masked = make_masked_image(x, mask, patch_size=model.patch_size).cpu()

                        # 3. reconstructed image (only masked patches replaced)
                        rec = apply_mae_reconstruction(x, pred, mask, patch_size=model.patch_size).cpu()

                        # 4. full decoder prediction (all patches replaced by prediction)
                        full_pred = unpatchify(pred, patch_size=model.patch_size).cpu()

                        # stack horizontally for each image
                        grid = torch.cat([orig, masked, rec, full_pred], dim=0)

                        writer.add_images(f"Reconstruction/step_{global_step}", grid, global_step)

                    model.train()

            

        avg = total_loss / len(loader)
        print(f'Epoch {epoch} avg loss {avg:.5f}')

        if LOGGING:
            writer.add_scalar('epoch/loss', avg, epoch)

        if (epoch + 1) % 5 == 0:    
            save_checkpoint(os.path.join(CKPT_DIR, f'mae_checkpoint_{epoch}.pth'), epoch, model, optimizer, scaler)
            # eval_acc = eval_knn(model, eval_train_loader, eval_test_loader, DEVICE)
            eval_acc = eval_linear_probe(model, eval_train_loader, eval_test_loader, DEVICE, lin_epochs=15)
            if LOGGING:
                # writer.add_scalar('eval/knn_acc', eval_acc, epoch)
                writer.add_scalar('eval/lp_acc', eval_acc, epoch)

    if LOGGING:
        writer.close()

if __name__ == '__main__':
    train_main()
