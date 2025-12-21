"""
DDPG Agent with Twin Q-Networks
"""
import torch
import torch.optim as optim
import numpy as np
from collections import deque

from ..models import Actor, Critic
from ..utils import AdaptiveOUNoise, PrioritizedReplayBuffer, AdaptiveLRScheduler


class DDPGAgent:
    """DDPG Agent with improvements"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.action_dim = config.ACTION_DIM
        
        # Actor
        self.actor = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.actor_target = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        
        # Critic
        self.critic = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.critic_target = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        
        # LR Scheduler
        self.lr_scheduler = AdaptiveLRScheduler(
            self.actor_optimizer, self.critic_optimizer,
            patience=config.LR_PATIENCE, factor=config.LR_FACTOR, min_lr=config.LR_MIN
        )
        
        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.BUFFER_CAPACITY,
            alpha=config.PRIORITIZED_REPLAY_ALPHA
        )
        
        # Noise
        self.noise = AdaptiveOUNoise(
            action_dim=config.ACTION_DIM,
            theta=config.OU_THETA,
            sigma_start=config.OU_SIGMA_START,
            sigma_end=config.OU_SIGMA_END,
            decay_steps=config.EXPLORATION_STEPS
        )
        
        # Exploration
        self.total_steps = 0
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        
        # Tracking
        self.recent_actions = deque(maxlen=50)
        
    def select_action(self, state, add_noise=True, force_explore=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            policy_action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        
        if not add_noise:
            return policy_action
        
        if force_explore or (np.random.rand() < self.epsilon):
            base_action = np.random.uniform(0.15, 0.85, self.action_dim)
        else:
            base_action = policy_action
        
        noise = self.noise.sample()
        action = base_action + noise
        action = np.clip(action, 0.0, 1.0)
        
        self.recent_actions.append(action.copy())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        if self.replay_buffer.size() < batch_size:
            return None, None
        
        sample = self.replay_buffer.sample(batch_size, beta=self.config.PRIORITIZED_REPLAY_BETA)
        if sample is None:
            return None, None
            
        states, actions, rewards, next_states, dones, weights, indices = sample
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        td_errors = (current_q1 - target_q).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        critic_loss = (weights * (current_q1 - target_q) ** 2).mean() + \
                      (weights * (current_q2 - target_q) ** 2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def update_learning_rate(self, episode_reward):
        self.lr_scheduler.step(episode_reward)
        
    def get_current_lr(self):
        return self.lr_scheduler.get_current_lrs()
    
    def get_action_diversity(self):
        if len(self.recent_actions) < 2:
            return 0.0
        actions = np.array(list(self.recent_actions))
        return np.mean(np.std(actions, axis=0))
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.epsilon = checkpoint.get('epsilon', 1.0)
