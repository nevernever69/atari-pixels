import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from latent_action_model import ActionToLatentMLP
from latent_action_data import get_action_latent_dataloaders

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"action_to_latent_{epoch:03d}.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "action_to_latent_best.pt"))

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        actions, latents = batch
        actions, latents = actions.to(device), latents.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = model(actions)  # (B, 35, 256)
            loss = criterion(logits.view(-1, 256), latents.view(-1))
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        running_loss += loss.item() * actions.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == latents).sum().item()
        total += latents.numel()
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            actions, latents = batch
            actions, latents = actions.to(device), latents.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(actions)
                loss = criterion(logits.view(-1, 256), latents.view(-1))
            running_loss += loss.item() * actions.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == latents).sum().item()
            total += latents.numel()
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    use_amp = device.type == 'cuda'
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'latent_action')
    
    train_loader, val_loader = get_action_latent_dataloaders(
        args.json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_actions=9
    )
    print(f"Train dataset: {len(train_loader.dataset)} pairs")
    print(f"Val dataset: {len(val_loader.dataset)} pairs")
    
    model = ActionToLatentMLP(input_dim=9, latent_dim=35, codebook_size=256)
    model = model.to(device)
    if use_amp:
        try:
            model = torch.compile(model, backend='inductor')
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }, is_best=(val_acc > best_val_acc), checkpoint_dir=checkpoint_dir, epoch=epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"[Checkpoint] Saved best model at epoch {epoch} with val acc {val_acc:.4f}")
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Action-to-Latent Mapping Model for Enduro')
    parser.add_argument('--json_path', type=str, default='/scratch/users/axb2032/world_model/data/actions/action_latent_pairs.json')
    parser.add_argument('--checkpoint_dir', type=str, default='/scratch/users/axb2032/world_model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
