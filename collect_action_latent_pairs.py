import os
import json
import torch
import numpy as np
from tqdm import tqdm
from atari_env import AtariEnv
from random_agent import RandomAgent
from latent_action_model import load_latent_action_model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def flatten_latent_indices(indices):
    if indices.ndim == 2:
        return indices.flatten().tolist()
    elif indices.ndim == 3:
        return [x.flatten().tolist() for x in indices]
    else:
        raise ValueError(f"Unexpected indices shape: {indices.shape}")

def collect_action_latent_pairs(
    out_path='/scratch/users/axb2032/world_model/data/actions/action_latent_pairs.json',
    n_pairs=100000,
    max_steps_per_episode=1000,
    seed=42
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = get_device()
    model, _ = load_latent_action_model('/scratch/users/axb2032/world_model/checkpoints/latent_action/best.pt', device)
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        try:
            model = torch.compile(model, backend='inductor')
        except Exception:
            pass
    env = AtariEnv(game_name="Enduro", return_rgb=True, grayscale=True)
    n_actions = env.action_space.n
    agent = RandomAgent(n_actions)
    collected = []
    np.random.seed(seed)
    torch.manual_seed(seed)
    episode = 0
    pbar = tqdm(total=n_pairs, desc='Collecting (action, latent_code) pairs')
    try:
        while len(collected) < n_pairs:
            obs, _ = env.reset()
            frame_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 210, 160)
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode and len(collected) < n_pairs:
                action = agent.select_action()
                next_obs, reward, terminated, truncated, info = env.step(action)
                frame_tp1 = torch.from_numpy(next_obs).float().unsqueeze(0).unsqueeze(0) / 255.0
                try:
                    with torch.no_grad():
                        frame_t = frame_t.to(device)
                        frame_tp1 = frame_tp1.to(device)
                        _, indices, *_ = model(frame_t, frame_tp1)
                except Exception as e:
                    print(f"Error during model call: {e}")
                    print(f"frame_t shape: {frame_t.shape}, dtype: {frame_t.dtype}")
                    print(f"frame_tp1 shape: {frame_tp1.shape}, dtype: {frame_tp1.dtype}")
                    raise
                latent_code = flatten_latent_indices(indices.cpu().squeeze(0))
                collected.append({
                    'action': int(action),
                    'latent_code': latent_code
                })
                pbar.update(1)
                frame_t = frame_tp1.cpu()
                steps += 1
                if terminated or truncated:
                    done = True
            episode += 1
    except KeyboardInterrupt:
        print('Interrupted. Saving collected data...')
    finally:
        env.close()
        pbar.close()
        with open(out_path, 'w') as f:
            json.dump(collected, f, indent=2)
        print(f"Saved {len(collected)} pairs to {out_path}")
        actions = [d['action'] for d in collected]
        print("Action distribution:", {a: actions.count(a) for a in set(actions)})
        print("Example latent code:", collected[0]['latent_code'] if collected else None)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect (action, latent_code) pairs for latent action mapping.')
    parser.add_argument('--out', type=str, default='/scratch/users/axb2032/world_model/data/actions/action_latent_pairs.json')
    parser.add_argument('--n_pairs', type=int, default=100000)
    parser.add_argument('--max_steps_per_episode', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    collect_action_latent_pairs(
        out_path=args.out,
        n_pairs=args.n_pairs,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed
    )
