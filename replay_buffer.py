import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    """Simple replay buffer with GPU storage."""
    
    def __init__(self, capacity, state_shape, device=torch.device('cuda:1')):
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.uint8, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = torch.as_tensor(state, device=self.device)
        self.actions[idx] = torch.as_tensor(action, device=self.device)
        self.rewards[idx] = torch.as_tensor(reward, device=self.device)
        self.next_states[idx] = torch.as_tensor(next_state, device=self.device)
        self.dones[idx] = torch.as_tensor(done, device=self.device)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, mode='normal', alpha=0.5):
        indices = np.random.randint(0, self.size, batch_size)
        states = self.states[indices].cpu().numpy()
        actions = self.actions[indices].cpu().numpy()
        rewards = self.rewards[indices].cpu().numpy()
        next_states = self.next_states[indices].cpu().numpy()
        dones = self.dones[indices].cpu().numpy()
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size

class PrioritizedReplayBuffer:
    """Prioritized replay buffer (CPU-based for simplicity)."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, mode='normal', alpha=0.5):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.stack(states), np.array(actions), np.array(rewards),
                np.stack(next_states), np.array(dones), weights, indices)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5
    
    def __len__(self):
        return self.size
