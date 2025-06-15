import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any

class AtariEnv:
    """
    Wrapper for Atari environment with preprocessing.
    If return_rgb=True, returns original RGB frames (210, 160, 3) or grayscale (210, 160).
    """
    
    def __init__(self, game_name: str = "Enduro", return_rgb: bool = False, grayscale: bool = False):
        """Initialize the environment with specific settings.
        Args:
            game_name: Name of the Atari game (e.g., 'Enduro', 'Breakout').
            return_rgb: If True, return original RGB or grayscale frames in reset/step.
            grayscale: If True, convert frames to grayscale even when return_rgb is True.
        """
        self.game_name = game_name
        self.grayscale = grayscale
        # Create the base Atari environment
        self.env = gym.make(
            f"ALE/{game_name}-v5",
            render_mode='rgb_array',
            frameskip=4,
            repeat_action_probability=0.0,
            full_action_space=False
        )
        
        self.action_space = self.env.action_space
        print(f"Action space: {self.action_space}")
        print(f"Action meanings: {self.env.unwrapped.get_action_meanings()}")
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84) if not return_rgb else (210, 160), dtype=np.uint8
        )
        
        self.lives = 0  # Enduro doesn't use lives; placeholder
        self.was_real_done = True
        self.living_penalty = -0.005
        self.life_loss_penalty = 0.0
        self.return_rgb = return_rgb
    
    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert RGB observation to 84x84 grayscale or return original."""
        if self.return_rgb:
            if self.grayscale:
                gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                return gray  # (210, 160)
            return obs  # (210, 160, 3)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset()
        self.lives = info.get('lives', 0)
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get('lives', 0)
        life_loss_reward = 0.0
        if lives < self.lives:
            life_loss_reward = self.life_loss_penalty
            self.lives = lives
        shaped_reward = reward + self.living_penalty + life_loss_reward
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, shaped_reward, terminated, truncated, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
