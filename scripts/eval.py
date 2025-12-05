import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from sklearn.neighbors import KNeighborsClassifier
from models.mae import MAE         # our MAE class

# --- Added path and imports for model configuration ---
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_CONFIGS, DEFAULT_MODEL_SIZE
# ------------------------------------------------------

# Eval dataloaders 
def build_eval_dataloaders(img_size):
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./eval_data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./eval_data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader



def extract_features(mae_model, imgs, pool="mean"):
    """
    Runs only MAE encoder forward (no decoder). Disable grad while calling this.
    """
    # Patch embed
    x = mae_model.patch_embed(imgs)
    x = x + mae_model.encoder_pos

    # Transformer blocks
    for blk in mae_model.enc_blocks:
        x = blk(x)

    x = mae_model.enc_norm(x)

    # Pooling
    if pool == "mean":
        feat = x.mean(dim=1)
    # no cls token in ths mae
    # elif pool == "cls":
    #     feat = x[:, 0]

    # Normalize
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat



def eval_knn(mae_model, train_loader, test_loader, device, pool="mean"):
    mae_model.eval()
    for p in mae_model.parameters():
        p.requires_grad = False

    print("Extracting train features...")
    train_feats, train_labels = [], []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        feats = extract_features(mae_model, imgs, pool)
        train_feats.append(feats.cpu())
        train_labels.append(labels)

    train_feats = torch.cat(train_feats).numpy()
    train_labels = torch.cat(train_labels).numpy()

    print("Training kNN...")
    knn = KNeighborsClassifier(n_neighbors=55, n_jobs=-1)
    knn.fit(train_feats, train_labels)

    print("Extracting test features...")
    test_feats, test_labels = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        feats = extract_features(mae_model, imgs, pool)
        test_feats.append(feats.cpu())
        test_labels.append(labels)

    test_feats = torch.cat(test_feats).numpy()
    test_labels = torch.cat(test_labels).numpy()

    acc = knn.score(test_feats, test_labels)
    print(f"★ k-NN accuracy: {acc:.4f}")
    for param in mae_model.parameters():
        param.requires_grad = True
    mae_model.train()
    return acc


def eval_linear_probe(mae_model, train_loader, test_loader, device,
                      pool="mean", lin_epochs=50):
    mae_model.eval()
    for p in mae_model.parameters():
        p.requires_grad = False

    num_classes = 10
    feat_dim = mae_model.enc_dim  # encoder hidden dim

    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.005)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training linear classifier on device: {device}...")


    for epoch in range(lin_epochs):
        classifier.train()
        total, correct, total_loss = 0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = extract_features(mae_model, imgs, pool)

            logits = classifier(feats)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        print(f"Epoch {epoch+1}/{lin_epochs}: "
              f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    
    classifier.eval()
    correct, total = 0, 0

    print("Testing linear classifier...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = extract_features(mae_model, imgs, pool)
            logits = classifier(feats)
            preds = logits.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"★ Linear probe accuracy: {acc:.4f}")
    for param in mae_model.parameters():
        param.requires_grad = True
    mae_model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MAE checkpoint")
    parser.add_argument("--method", type=str, default="knn", choices=["knn", "linear"])
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--lin_epochs", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=96)
    # --- New argument for model size ---
    parser.add_argument('--model_size', type=str, default=DEFAULT_MODEL_SIZE, 
                        choices=list(MODEL_CONFIGS.keys()), help='Model configuration size for checkpoint loading')
    # -----------------------------------

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # --- Select Model Configuration ---
    cfg = MODEL_CONFIGS[args.model_size]
    # ----------------------------------


    print("Loading MAE checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device)

    # --- Initialize MAE with selected configuration ---
    mae_model = MAE(
        img_size=args.img_size,
        patch_size=cfg['PATCH_SIZE'],
        enc_dim=cfg['ENC_DIM'],
        enc_depth=cfg['ENC_DEPTH'],
        enc_heads=cfg['ENC_HEADS'],
        dec_dim=cfg['DEC_DIM'],
        dec_depth=cfg['DEC_DEPTH'],
        dec_heads=cfg['DEC_HEADS'],
        mask_ratio=cfg['MASK_RATIO'],
    ).to(device)
    # --------------------------------------------------

    mae_model.load_state_dict(ckpt["model"], strict=True)
    print("Loaded MAE.")


    train_loader, test_loader = build_eval_dataloaders(args.img_size)

    if args.method == "knn":
        eval_knn(mae_model, train_loader, test_loader, device, args.pool)
    else:
        eval_linear_probe(mae_model, train_loader, test_loader,
                          device, args.pool, args.lin_epochs)


if __name__ == "__main__":
    main()