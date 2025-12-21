"""
Training Script for HVAC DDPG Control
Supports both forecast and no-forecast modes

Usage:
    # Train with forecast (state_dim=15)
    python scripts/train.py --forecast
    
    # Train without forecast (state_dim=14)
    python scripts/train.py --no-forecast
    
    # Custom settings
    python scripts/train.py --forecast --episodes 100 --batch-size 512 --verbose
"""

import sys
import os
import json
import numpy as np
import random
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_manager import get_train_config
from src.agents import DDPGAgent
from src.environments import HVACEnvironment
from src.visualization import plot_training_progress


def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train HVAC DDPG Agent')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--forecast', action='store_true',
                           help='Use temperature forecast (state_dim=15)')
    mode_group.add_argument('--no-forecast', action='store_true',
                           help='No temperature forecast (state_dim=14)')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes (default: 50)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for training (default: 512)')
    parser.add_argument('--save-freq', type=int, default=2,
                       help='Save checkpoint every N episodes (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Display options
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed step information')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting training progress')
    
    return parser.parse_args()


def train_ddpg(config, verbose=False):
    """
    Train DDPG agent on HVAC environment
    
    Args:
        config: Configuration object
        verbose: Print detailed information
        
    Returns:
        agent: Trained agent
        episode_rewards: List of episode rewards
        episode_stats: List of episode statistics
    """
    # Set random seeds
    set_random_seeds(config.RANDOM_SEED)
    
    # Create directories
    config.create_directories()
    
    # Display configuration
    config.display()
    
    # Initialize environment and agent
    env = HVACEnvironment(config, verbose=False)
    agent = DDPGAgent(config)
    
    print(f"\n{'='*70}")
    print(f"{'STARTING TRAINING':^70}")
    print(f"{'='*70}\n")
    
    # Training tracking
    episode_rewards = []
    episode_stats = []
    best_reward = -np.inf
    
    for episode in range(1, config.NUM_EPISODES + 1):
        print(f"\n{'='*70}")
        print(f"Episode {episode}/{config.NUM_EPISODES}")
        print(f"{'='*70}")
        
        # Reset environment
        state = env.reset(episode=episode)
        done = False
        step_count = 0
        episode_reward = 0
        
        # Force exploration in early episodes
        force_explore = episode <= 3
        
        while not done:
            # Select action
            action = agent.select_action(
                state, 
                add_noise=True, 
                force_explore=force_explore
            )
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            actor_loss, critic_loss = agent.train()
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Verbose output
            if verbose and step_count % 100 == 0 and info:
                print(f"  Step {step_count:04d} | "
                      f"T={info['T_zone']:.2f}°C | "
                      f"RH={info['RH_zone']:.3f} | "
                      f"P={info['P_total']/1000:.2f}kW | "
                      f"r={reward:.3f}")
        
        # Episode statistics
        stats = env.get_episode_stats()
        episode_rewards.append(episode_reward)
        episode_stats.append(stats)
        
        # Update learning rate
        agent.update_learning_rate(episode_reward)
        lrs = agent.get_current_lr()
        
        # Print episode summary
        print(f"\n📊 Episode Summary:")
        print(f"   Total Reward:      {episode_reward:.2f}")
        print(f"   Steps:             {step_count}")
        print(f"   Avg Reward/Step:   {episode_reward/step_count:.4f}")
        print(f"   Avg Temperature:   {stats['avg_T']:.2f}°C ± {stats['std_T']:.2f}°C")
        print(f"   Avg Humidity:      {stats['avg_RH']:.3f} ± {stats['std_RH']:.3f}")
        print(f"   T Comfort Ratio:   {stats['T_comfort_ratio']*100:.1f}%")
        print(f"   RH Comfort Ratio:  {stats['RH_comfort_ratio']*100:.1f}%")
        print(f"   Action Diversity:  {stats['action_diversity']:.4f}")
        print(f"   Buffer Size:       {agent.replay_buffer.size()}")
        print(f"   Epsilon:           {agent.epsilon:.4f}")
        print(f"   Actor LR:          {lrs['actor_lr']:.2e}")
        print(f"   Critic LR:         {lrs['critic_lr']:.2e}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = config.CHECKPOINT_PATH / "best_model.pth"
            agent.save(best_path)
            print(f"\n   🎯 New best reward! Saved to {best_path}")
        
        # Periodic save
        if episode % config.SAVE_FREQ == 0:
            checkpoint_path = config.CHECKPOINT_PATH / f"checkpoint_ep{episode}.pth"
            agent.save(checkpoint_path)
            print(f"   💾 Checkpoint saved to {checkpoint_path}")
    
    # Final save
    final_path = config.CHECKPOINT_PATH / "final_model.pth"
    agent.save(final_path)
    print(f"\n✅ Training completed! Final model saved to {final_path}")
    
    # Save training history
    history = {
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_stats': episode_stats,
        'config': {
            'use_forecast': config.USE_FORECAST,
            'state_dim': config.STATE_DIM,
            'action_dim': config.ACTION_DIM,
            'num_episodes': config.NUM_EPISODES,
            'lr_actor': config.LR_ACTOR,
            'lr_critic': config.LR_CRITIC,
            'gamma': config.GAMMA,
            'tau': config.TAU
        }
    }
    
    history_path = config.RESULTS_PATH / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📝 Training history saved to {history_path}")
    
    return agent, episode_rewards, episode_stats


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine mode
    use_forecast = args.forecast
    mode_str = "WITH FORECAST" if use_forecast else "NO FORECAST"
    
    print(f"\n{'='*70}")
    print(f"HVAC DDPG TRAINING - {mode_str}".center(70))
    print(f"{'='*70}\n")
    
    # Get configuration
    config = get_train_config(use_forecast=use_forecast)
    
    # Apply command line arguments
    config.NUM_EPISODES = args.episodes
    config.BATCH_SIZE = args.batch_size
    config.SAVE_FREQ = args.save_freq
    config.RANDOM_SEED = args.seed
    
    # Train agent
    agent, rewards, stats = train_ddpg(config, verbose=args.verbose)
    
    # Plot training progress (unless disabled)
    if not args.no_plot:
        try:
            plot_training_progress(rewards, stats, config.RESULTS_PATH)
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE".center(70))
    print(f"{'='*70}")
    print(f"Mode:              {mode_str}")
    print(f"Total Episodes:    {len(rewards)}")
    print(f"Best Reward:       {max(rewards):.2f}")
    print(f"Final Reward:      {rewards[-1]:.2f}")
    print(f"Checkpoints:       {config.CHECKPOINT_PATH}")
    print(f"Results:           {config.RESULTS_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
