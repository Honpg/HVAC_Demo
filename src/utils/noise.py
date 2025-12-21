"""
Adaptive Ornstein-Uhlenbeck Noise
"""
import numpy as np


class AdaptiveOUNoise:
    """OU noise with exponential decay"""
    
    def __init__(self, action_dim, mu=0.0, theta=0.15,
                 sigma_start=0.7, sigma_end=0.15, decay_steps=100000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps
        self.current_step = 0
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def get_sigma(self):
        decay = min(1.0, self.current_step / self.decay_steps)
        return self.sigma_start * np.exp(-2.5 * decay) + self.sigma_end
    
    def sample(self):
        sigma = self.get_sigma()
        dx = self.theta * (self.mu - self.state) + sigma * np.random.randn(self.action_dim)
        self.state += dx
        self.current_step += 1
        return self.state
