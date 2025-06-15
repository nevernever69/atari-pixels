"""
Controls:
PERIOD (.) - No Operation (NOOP, default)
SPACE - Fire
RIGHT ARROW - Right
LEFT ARROW - Left
DOWN ARROW - Down
S - Down+Right
A - Down+Left
D - Right+Fire
W - Left+Fire
ESC or Q - Quit

Run:
python play_neural_enduro.py --temperature 0.01
"""

import os
import sys
import torch
import numpy as np
import pygame
from PIL import Image
import time
import argparse
import imageio
from latent_action_model import load_latent_action_model, ActionToLatentMLP

# Configuration
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 840
FPS = 15

def get_device():
    return torch.device('cpu')

def action_to_onehot(action_idx, n_actions, device):
    onehot = torch.zeros(1, n_actions, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def grayscale_to_rgb(frame):
    """Convert grayscale (1, H, W) or (H, W) to RGB (H, W, 3) for display."""
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame.squeeze(0)  # (H, W)
    elif frame.ndim != 2:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    frame = frame[..., np.newaxis].repeat(3, axis=-1)  # (H, W, 3)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Playable Neural Enduro")
    parser.add_argument('--temperature', type=float, default=0.01, help='Sampling temperature for latent prediction')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps before auto-quit')
    args = parser.parse_args()
    
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Enduro")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    print("[INFO] Loading world model...")
    world_model, step = load_latent_action_model('/run/media/never/new2/rudoro/latent_action/best.pt', device)
    world_model.to(device)
    world_model.eval()
    
    print("[INFO] Loading action-to-latent model...")
    action_model = ActionToLatentMLP(input_dim=9, latent_dim=35, codebook_size=256).to(device)
    ckpt_path = '/run/media/never/new2/rudoro/latent_action/action_to_latent_best.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    action_model.load_state_dict(cleaned_state_dict)
    action_model.eval()
    
    print("[INFO] Loading initial frame...")
    init_frame_path = '/run/media/never/new2/rudoro/data/0.png'
    if not os.path.exists(init_frame_path):
        raise FileNotFoundError(f"Initial frame {init_frame_path} not found. Generate it using generate_initial_frame.py.")
    init_img = Image.open(init_frame_path).convert('L')  # Grayscale
    init_frame_np = np.array(init_img, dtype=np.float32) / 255.0  # (210, 160)
    current_frame = torch.from_numpy(init_frame_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 210, 160)
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT', 'RIGHTFIRE', 'LEFTFIRE']
    key_to_action = {
        pygame.K_PERIOD: 0,  # NOOP
        pygame.K_SPACE: 1,   # FIRE
        pygame.K_RIGHT: 2,   # RIGHT
        pygame.K_LEFT: 3,    # LEFT
        pygame.K_DOWN: 4,    # DOWN
        pygame.K_s: 5,       # DOWNRIGHT
        pygame.K_a: 6,       # DOWNLEFT
        pygame.K_d: 7,       # RIGHTFIRE
        pygame.K_w: 8        # LEFTFIRE
    }
    
    print("\nNeural Enduro Controls:")
    print("----------------------")
    print("PERIOD (.) - No Operation (NOOP, default)")
    print("SPACE - Fire")
    print("RIGHT ARROW - Right")
    print("LEFT ARROW - Left")
    print("DOWN ARROW - Down")
    print("S - Down+Right")
    print("A - Down+Left")
    print("D - Right+Fire")
    print("W - Left+Fire")
    print("ESC or Q - Quit\n")
    
    action_idx = 0
    last_displayed_action = ""
    step = 0
    frames = []
    
    running = True
    while running and step < args.steps:
        action_idx = 0  # Default to NOOP
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in key_to_action:
                    action_idx = key_to_action[event.key]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action_idx = 1
            last_displayed_action = "FIRE"
        elif keys[pygame.K_RIGHT]:
            action_idx = 2
            last_displayed_action = "RIGHT"
        elif keys[pygame.K_LEFT]:
            action_idx = 3
            last_displayed_action = "LEFT"
        elif keys[pygame.K_DOWN]:
            action_idx = 4
            last_displayed_action = "DOWN"
        elif keys[pygame.K_s]:
            action_idx = 5
            last_displayed_action = "DOWNRIGHT"
        elif keys[pygame.K_a]:
            action_idx = 6
            last_displayed_action = "DOWNLEFT"
        elif keys[pygame.K_d]:
            action_idx = 7
            last_displayed_action = "RIGHTFIRE"
        elif keys[pygame.K_w]:
            action_idx = 8
            last_displayed_action = "LEFTFIRE"
        elif keys[pygame.K_PERIOD]:
            action_idx = 0
            last_displayed_action = "NOOP"
        
        if action_idx != 0:
            print(f"Action: {action_names[action_idx]}")
        
        with torch.no_grad():
            onehot = action_to_onehot(action_idx, n_actions=9, device=device)
            logits = action_model(onehot)  # (1, 35, 256)
            indices = action_model.sample_latents(logits, temperature=args.temperature)  # (1, 35)
            indices = indices.view(1, 5, 7)  # (1, 5, 7)
            embeddings = world_model.vq.embeddings
            indices = indices.to(embeddings.weight.device)
            quantized = embeddings(indices)  # (1, 5, 7, 128)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (1, 128, 5, 7)
            frame_in = current_frame.permute(0, 1, 3, 2)  # (1, 1, 210, 160) -> (1, 1, 160, 210)
            next_frame = world_model.decoder(quantized, frame_in)  # (1, 1, 160, 210)
            next_frame = next_frame.permute(0, 1, 3, 2)  # (1, 1, 210, 160)
            current_frame = next_frame.clone()
        
        frame_np = current_frame.squeeze(0).cpu().numpy()  # (1, 210, 160) -> (210, 160)
        frame_rgb = grayscale_to_rgb(frame_np)  # (210, 160, 3)
        frame_rgb = (frame_rgb * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame_rgb.copy())
        
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        scaled_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT - 40))
        
        window.fill((0, 0, 0))
        window.blit(scaled_surface, (0, 0))
        
        step += 1
        info_text = f"Step: {step}"
        if last_displayed_action:
            info_text += f" | Last Action: {last_displayed_action}"
        info_text += f" | Temperature: {args.temperature}"
        
        text_surface = font.render(info_text, True, (255, 255, 255))
        window.blit(text_surface, (10, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    print("[INFO] Saving video...")
    os.makedirs('/run/media/never/new2/rudoro/data', exist_ok=True)
    imageio.mimsave('/run/media/never/new2/rudoro/data/neural_enduro_game.gif', frames, fps=15)
    print(f"Saved video to /run/media/never/new2/rudoro/data/neural_enduro_game.gif")
    print("Game closed.")

if __name__ == "__main__":
    main()
