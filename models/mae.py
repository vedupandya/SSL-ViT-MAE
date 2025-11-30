import torch
import torch.nn as nn
from .vit_blocks import PatchEmbed, Block
from utils.vision import patchify

class MAE(nn.Module):
    def __init__(self, img_size=96, patch_size=16, enc_dim=384, enc_depth=12, enc_heads=6,
                 dec_dim=192, dec_depth=4, dec_heads=6, mask_ratio=0.25):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_dim)
        self.num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio

        self.encoder_pos = nn.Parameter(torch.randn(1, self.num_patches, enc_dim))
        self.enc_blocks = nn.ModuleList([Block(enc_dim, enc_heads) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos = nn.Parameter(torch.randn(1, self.num_patches, dec_dim))

        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, patch_size * patch_size * 3)

        self.patch_size = patch_size

    def random_masking(self, x, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B, N, C = x.shape
        len_keep = int(N * (1.0 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))
        mask = torch.ones([B, N], device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return x_masked, mask, ids_keep, ids_restore

    def forward_encoder(self, imgs):
        x = self.patch_embed(imgs)
        x = x + self.encoder_pos
        x_masked, mask, ids_keep, ids_restore = self.random_masking(x, None)
        for blk in self.enc_blocks:
            x_masked = blk(x_masked)
        latent = self.enc_norm(x_masked)
        return latent, mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        dec = self.decoder_embed(latent)
        B, N = ids_restore.shape
        D = dec.size(-1)
        len_keep = dec.shape[1]
        len_mask = N - len_keep
        mask_tokens = self.mask_token.expand(B, len_mask, D)
        x_ = torch.cat([dec, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_ = x_ + self.decoder_pos
        for blk in self.dec_blocks:
            x_ = blk(x_)
        x_ = self.dec_norm(x_)
        pred = self.dec_pred(x_)
        return pred

    def forward(self, imgs):
        imgs = imgs.float()
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask

def mae_loss(pred, imgs, patch_size):
    patches = patchify(imgs, patch_size)
    loss_per_patch = ((pred - patches) ** 2).mean(dim=-1)
    return loss_per_patch