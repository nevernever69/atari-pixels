#!/usr/bin/env python3
import glob
import os
import math
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from datetime import datetime

from latent_action_data import AtariFramePairDataset
from latent_action_model import LatentActionVQVAE, load_latent_action_model

# Utility Functions
def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean(dim=[2,3], keepdim=True)
    mu2 = img2.mean(dim=[2,3], keepdim=True)
    sigma1 = ((img1 - mu1) ** 2).mean(dim=[2,3], keepdim=True)
    sigma2 = ((img2 - mu2) ** 2).mean(dim=[2,3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2,3], keepdim=True)
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean().item()

def codebook_entropy(indices, codebook_size):
    hist = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    prob = hist / hist.sum()
    entropy = -(prob[prob > 0] * prob[prob > 0].log()).sum().item()
    return entropy, hist.cpu().numpy()

def save_checkpoint(state, is_best, checkpoint_dir, step):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{step:06d}.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pt"))

def check_latent_encoding(model, val_loader, device, step, output_dir=None, use_wandb=False):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    with torch.no_grad():
        frame_t, frame_tp1 = next(iter(val_loader))
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
        frame_diff = torch.abs(frame_tp1 - frame_t)
        car_mask = (frame_diff > 0.05).float()
        recon, indices, commit_loss, codebook_loss = model(frame_t, frame_tp1)
        x = torch.cat([frame_t, frame_tp1], dim=1)
        z = model.encoder(x)
        for i in range(min(4, frame_t.size(0))):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(frame_tp1[i, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title('Frame with Car')
            axes[1].imshow(car_mask[i, 0].cpu().numpy(), cmap='hot')
            axes[1].set_title('Car Position')
            latent_vis = z[i].mean(0).cpu().numpy()
            latent_vis = (latent_vis - latent_vis.min()) / (latent_vis.max() - latent_vis.min() + 1e-8)
            axes[2].imshow(latent_vis, cmap='viridis')
            axes[2].set_title('Latent Space Activation')
            plt.tight_layout()
            if output_dir:
                filename = os.path.join(output_dir, f'car_encoding_ex{i}_step{step}.png')
                fig.savefig(filename)
                print(f"Saved {filename}")
            plt.close(fig)

def test_encoder_decoder(model, val_loader, device, step, output_dir=None, use_wandb=False):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    with torch.no_grad():
        frame_t, frame_tp1 = next(iter(val_loader))
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
        recon, indices, _, _ = model(frame_t, frame_tp1)
        frame_t_permuted = frame_t.permute(0, 1, 3, 2)
        frame_tp1_permuted = frame_tp1.permute(0, 1, 3, 2)
        x = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)
        z = model.encoder(x)
        frame_diff = torch.abs(frame_tp1 - frame_t)
        car_mask = (frame_diff > 0.05).float()
        car_mask_latent = F.interpolate(car_mask, size=z.shape[2:], mode='nearest')
        z_enhanced = z.clone()
        z_enhanced = z_enhanced + 20.0 * car_mask_latent.expand_as(z_enhanced)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, model.vq.embedding_dim)
        z_enhanced_flat = z_enhanced.permute(0, 2, 3, 1).contiguous().view(-1, model.vq.embedding_dim)
        d_original = (z_flat.pow(2).sum(1, keepdim=True)
                     - 2 * z_flat @ model.vq.embeddings.weight.t()
                     + model.vq.embeddings.weight.pow(2).sum(1))
        indices_original = torch.argmin(d_original, dim=1)
        d_enhanced = (z_enhanced_flat.pow(2).sum(1, keepdim=True)
                     - 2 * z_enhanced_flat @ model.vq.embeddings.weight.t()
                     + model.vq.embeddings.weight.pow(2).sum(1))
        indices_enhanced = torch.argmin(d_enhanced, dim=1)
        indices_original = indices_original.view(z.shape[0], z.shape[2], z.shape[3])
        indices_enhanced = indices_enhanced.view(z.shape[0], z.shape[2], z.shape[3])
        code_changes = (indices_original != indices_enhanced).float()
        quantized_enhanced, indices_enhanced_out, _, _ = model.vq(z_enhanced)
        recon_permuted_enhanced = model.decoder(quantized_enhanced, frame_t_permuted)
        recon_enhanced = recon_permuted_enhanced.permute(0, 1, 3, 2)
        for i in range(min(4, frame_t.size(0))):
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            def normalize_for_display(img_tensor):
                img_np = img_tensor[0].cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                return img_np
            axes[0].imshow(normalize_for_display(frame_tp1[i]), cmap='gray')
            axes[0].set_title('Target Frame')
            axes[1].imshow(normalize_for_display(recon[i]), cmap='gray')
            axes[1].set_title('Normal Reconstruction')
            axes[2].imshow(normalize_for_display(recon_enhanced[i]), cmap='gray')
            axes[2].set_title('Enhanced Reconstruction')
            diff = torch.abs(recon_enhanced[i] - recon[i])[0].cpu().numpy()
            diff_max = diff.max()
            if diff_max > 1e-6:
                diff = diff / diff_max
            else:
                diff = np.zeros_like(diff)
            axes[3].imshow(diff, cmap='hot')
            axes[3].set_title('Reconstruction Difference')
            axes[4].imshow(code_changes[i].cpu().numpy(), cmap='hot')
            axes[4].set_title('Codebook Changes')
            plt.tight_layout()
            if output_dir:
                filename = os.path.join(output_dir, f'latent_manipulation_ex{i}_step{step}.png')
                fig.savefig(filename)
                print(f"Saved {filename}")
            plt.close(fig)

def print_model_structure(model):
    state_dict = model.state_dict()
    print(f"Model has {len(state_dict.keys())} parameters")
    for i, key in enumerate(sorted(state_dict.keys())):
        if i < 10 or i > len(state_dict.keys()) - 10:
            print(f"  {key}: {state_dict[key].shape}")
        elif i == 10:
            print("  ...")

def analyze_checkpoint(checkpoint_path, data_dir, output_dir, device=None, grayscale=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    model, step = load_latent_action_model(checkpoint_path, device)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from step {step}")
    val_set = AtariFramePairDataset(data_dir, split='val', grayscale=grayscale)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=2)
    print("Running latent encoding analysis...")
    check_latent_encoding(model, val_loader, device, step,
                          output_dir=os.path.join(output_dir, 'latent_encoding'),
                          use_wandb=False)
    print("Running encoder-decoder test...")
    test_encoder_decoder(model, val_loader, device, step,
                         output_dir=os.path.join(output_dir, 'encoder_decoder_test'),
                         use_wandb=False)
    print(f"Analysis complete. Results saved to {output_dir}")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    compile_mode = "inductor" if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    pin_memory = device.type == 'cuda'
    train_set = AtariFramePairDataset(args.data_dir, split='train', grayscale=args.grayscale)
    val_set = AtariFramePairDataset(args.data_dir, split='val', grayscale=args.grayscale)
    test_set = AtariFramePairDataset(args.data_dir, split='test', grayscale=args.grayscale)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    def get_stats(dataset, name):
        n_episodes = len(dataset.episode_dirs)
        n_pairs = len(dataset)
        sample = dataset[0]
        shape = sample[0].shape
        print(f"{name}: {n_episodes} episodes, {n_pairs} pairs, frame shape: {shape}")
    print("--- Data Statistics ---")
    get_stats(train_set, "Train")
    get_stats(val_set, "Val")
    get_stats(test_set, "Test")
    print("-----------------------")
    model = LatentActionVQVAE(grayscale=args.grayscale)
    model = model.to(device)
    if compile_mode:
        try:
            print(f"Compiling model with {compile_mode} backend...")
            model = torch.compile(model, backend=compile_mode)
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation")
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=args.lr_min)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    best_val_loss = float('inf')
    global_step = 0
    grad_accum = args.grad_accum
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'latent_action')
    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    pbar = tqdm(range(args.max_iters), desc="Training", dynamic_ncols=True)
    while global_step < args.max_iters:
        try:
            frame_t, frame_tp1 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            frame_t, frame_tp1 = next(data_iter)
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            recon, indices, commit_loss, codebook_loss = model(frame_t, frame_tp1)
            frame_diff = torch.abs(frame_tp1 - frame_t)
            motion_weight = 1.0 + 10.0 * (frame_diff > 0.05).float()
            rec_loss = (motion_weight * (recon - frame_tp1)**2).mean()
            entropy, hist = codebook_entropy(indices, 256)
            entropy_reg = -args.entropy_weight * entropy
            loss = rec_loss + commit_loss + codebook_loss + entropy_reg
        if scaler:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()
        if (global_step + 1) % grad_accum == 0:
            if args.grad_clip > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if (global_step + 1) % args.log_interval == 0:
            with torch.no_grad():
                psnr_val = psnr(recon, frame_tp1)
                ssim_val = ssim(recon, frame_tp1)
                l1 = F.l1_loss(recon, frame_tp1).item()
                l2 = F.mse_loss(recon, frame_tp1).item()
                print(f"Step {global_step+1}: Loss={loss.item():.4f}, Rec={rec_loss.item():.4f}, "
                      f"Commit={commit_loss.item():.4f}, Codebook={codebook_loss.item():.4f}, "
                      f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}, L1={l1:.4f}, L2={l2:.4f}, "
                      f"Entropy={entropy:.2f}")
        if (global_step + 1) % args.recon_interval == 0:
            with torch.no_grad():
                recon = torch.clamp(recon, 0, 1)  # Clip to [0, 1]
                grid = make_grid(torch.cat([frame_t, frame_tp1, recon], dim=0), nrow=args.batch_size)
                os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
                plt.figure(figsize=(15, 5))
                plt.imshow(grid[0].cpu().numpy(), cmap='gray')  # Single channel for grayscale
                plt.title(f'Reconstructions Step {global_step+1}')
                plt.savefig(os.path.join(checkpoint_dir, f'recon_step_{global_step+1}.png'))
                plt.close()
        if (global_step + 1) % args.codebook_reset_interval == 0:
            with torch.no_grad():
                used = (hist > 0).sum()
                if used < 0.5 * 256:
                    print(f"Resetting unused codebook entries at step {global_step+1}")
                    model.vq.embeddings.weight.data[hist == 0] = torch.randn_like(model.vq.embeddings.weight.data[hist == 0])
        if (global_step + 1) % args.ckpt_interval == 0:
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': global_step + 1
            }, is_best=False, checkpoint_dir=checkpoint_dir, step=global_step + 1)
        if (global_step + 1) % args.diagnostic_interval == 0:
            print(f"Running diagnostics at step {global_step+1}")
            check_latent_encoding(
                model, val_loader, device, global_step + 1,
                output_dir=os.path.join(checkpoint_dir, 'latent_encoding'),
                use_wandb=False
            )
            test_encoder_decoder(
                model, val_loader, device, global_step + 1,
                output_dir=os.path.join(checkpoint_dir, 'encoder_decoder_test'),
                use_wandb=False
            )
        if (global_step + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_loader, device, scaler, pbar_desc='Val')
            print(f"Validation Loss at step {global_step+1}: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step + 1
                }, is_best=True, checkpoint_dir=checkpoint_dir, step=global_step + 1)
        global_step += 1
        pbar.update(1)
    pbar.close()
    test_loss = evaluate(model, test_loader, device, scaler, pbar_desc='Test')
    print(f"Test loss: {test_loss:.4f}")

def evaluate(model, loader, device, scaler, pbar_desc=None):
    model.eval()
    losses = []
    with torch.no_grad():
        if pbar_desc:
            loader_iter = tqdm(loader, desc=pbar_desc, dynamic_ncols=True)
        else:
            loader_iter = loader
        for frame_t, frame_tp1 in loader_iter:
            frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                recon, indices, commit_loss, codebook_loss = model(frame_t, frame_tp1)
                rec_loss = F.mse_loss(recon, frame_tp1)
                loss = rec_loss + commit_loss + codebook_loss
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='bulk_videos', help='Directory with episode folders')
    parser.add_argument('--checkpoint_dir', type=str, default='/scratch/users/axb2032/world_model', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grayscale', action='store_true', default=True)
    parser.add_argument('--entropy_weight', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--recon_interval', type=int, default=1000)
    parser.add_argument('--ckpt_interval', type=int, default=5000)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--codebook_reset_interval', type=int, default=2500)
    parser.add_argument('--analyze', action='store_true', help='Analyze latest checkpoint')
    parser.add_argument('--analysis_output_dir', type=str, default='/scratch/users/axb2032/world_model/analysis', help='Output directory for analysis')
    parser.add_argument('--diagnostic_interval', type=int, default=5000, help='Steps between diagnostic visualizations')
    args = parser.parse_args()

    if args.analyze:
        checkpoint_dir = os.path.join(args.checkpoint_dir, 'latent_action')
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pt')))
        if not checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, 'best.pt')
            if not os.path.exists(checkpoint_path):
                print("No checkpoints found!")
                exit(1)
        else:
            checkpoint_path = checkpoints[-1]
        analyze_checkpoint(
            checkpoint_path=checkpoint_path,
            data_dir=args.data_dir,
            output_dir=args.analysis_output_dir,
            device=None,
            grayscale=args.grayscale
        )
        exit(0)

    train(args)
