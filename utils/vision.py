import torch
import torch.nn.functional as F

# Convert image to 6×6 patches
def patchify(imgs, patch_size=16):
    """
    imgs: (B,3,H,W)
    return (B, N, patch_dim)
    """
    B, C, H, W = imgs.shape
    p = patch_size
    assert H % p == 0 and W % p == 0

    h = H // p
    w = W // p

    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    patches = x.reshape(B, h*w, C*p*p)
    return patches

def unpatchify(patches, patch_size=16, img_size=96):
    B, N, patch_dim = patches.shape
    C = 3
    H = W = img_size
    p = patch_size
    patches = patches.reshape(B, int(H/p), int(W/p), C, p, p)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    imgs = patches.reshape(B, C, H, W)
    return imgs


# Fill only the masked patches using predictions, full decoder output is in the training loop
def apply_mae_reconstruction(imgs, pred, mask, patch_size=16):
    """
    imgs: (B,3,H,W)
    pred: (B,N,patch_dim)
    mask: (B,N) 1 = masked, 0 = visible
    """
    B, C, H, W = imgs.shape
    p = patch_size
    N = pred.shape[1]

    # Patchify original images
    imgs_patch = patchify(imgs, patch_size=p)

    # Replace ONLY masked patches with predictions
    restored = imgs_patch.clone()
    restored[mask.bool()] = pred[mask.bool()]

    # Unpatchify restored patches
    restored_imgs = unpatchify(restored, patch_size=p, img_size=H)
    return restored_imgs


# Make a masked visualization (gray patches)
def make_masked_image(imgs, mask, patch_size=16):
    B, C, H, W = imgs.shape
    p = patch_size

    patches = patchify(imgs, patch_size=p)
    gray_patch = torch.zeros_like(patches[:, 0]) + 0.5  # gray

    patches_masked = patches.clone()
    patches_masked[mask.bool()] = gray_patch

    masked_imgs = unpatchify(patches_masked, patch_size=p, img_size=H)
    return masked_imgs
