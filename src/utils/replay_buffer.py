"""
Prioritized Experience Replay Buffer
"""
import numpy as np
from collections import deque


class PrioritizedReplayBuffer:
    """Prioritized replay with TD-error based sampling"""
    
    def __init__(self, capacity=200000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
            
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones), weights, indices
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def size(self):
        return len(self.buffer)
