import os
import numpy as np
from PIL import Image
from atari_env import AtariEnv

def generate_initial_frame(output_path='/run/media/never/new2/rudoro/data/0.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = AtariEnv(game_name="Enduro", return_rgb=True, grayscale=True)
    obs, _ = env.reset()  # (210, 160)
    Image.fromarray(obs).save(output_path)
    print(f"Saved initial frame to {output_path}")
    env.close()

if __name__ == "__main__":
    generate_initial_frame()
