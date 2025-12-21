"""
Adaptive Learning Rate Scheduler
"""
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AdaptiveLRScheduler:
    """Adaptive LR for Actor and Critic"""
    
    def __init__(self, actor_optimizer, critic_optimizer,
                 patience=5, factor=0.5, min_lr=1e-6, verbose=True):
        self.actor_scheduler = ReduceLROnPlateau(
            actor_optimizer, mode='max', factor=factor,
            patience=patience, min_lr=min_lr, verbose=False
        )
        self.critic_scheduler = ReduceLROnPlateau(
            critic_optimizer, mode='max', factor=factor,
            patience=patience, min_lr=min_lr, verbose=False
        )
        self.best_reward = -np.inf
        self.verbose = verbose
        self.actor_lr_history = []
        self.critic_lr_history = []
        
    def step(self, episode_reward):
        old_actor_lr = self.get_lr(self.actor_scheduler.optimizer)
        old_critic_lr = self.get_lr(self.critic_scheduler.optimizer)
        
        self.actor_scheduler.step(episode_reward)
        self.critic_scheduler.step(episode_reward)
        
        new_actor_lr = self.get_lr(self.actor_scheduler.optimizer)
        new_critic_lr = self.get_lr(self.critic_scheduler.optimizer)
        
        if self.verbose:
            if old_actor_lr != new_actor_lr:
                print(f"   📉 Actor LR: {old_actor_lr:.2e} → {new_actor_lr:.2e}")
            if old_critic_lr != new_critic_lr:
                print(f"   📉 Critic LR: {old_critic_lr:.2e} → {new_critic_lr:.2e}")
        
        self.actor_lr_history.append(new_actor_lr)
        self.critic_lr_history.append(new_critic_lr)
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
    
    def get_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']
    
    def get_current_lrs(self):
        return {
            'actor_lr': self.get_lr(self.actor_scheduler.optimizer),
            'critic_lr': self.get_lr(self.critic_scheduler.optimizer)
        }
