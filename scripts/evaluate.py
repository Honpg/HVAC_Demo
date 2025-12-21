"""
Evaluation Script for HVAC DDPG Control
Supports both forecast and no-forecast modes

Usage:
    # Evaluate with forecast model
    python scripts/evaluate.py --forecast --checkpoint best_model.pth
    
    # Evaluate without forecast model
    python scripts/evaluate.py --no-forecast --checkpoint best_model.pth
    
    # With verbose output
    python scripts/evaluate.py --forecast --checkpoint best_model.pth --verbose
    
    # Skip plots
    python scripts/evaluate.py --forecast --checkpoint best_model.pth --no-plot
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_manager import get_eval_config
from src.models import Actor
from src.environments import HVACEnvironment
from src.visualization import plot_evaluation_results, print_comfort_statistics


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained HVAC DDPG Agent')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--forecast', action='store_true',
                           help='Model with temperature forecast (state_dim=15)')
    mode_group.add_argument('--no-forecast', action='store_true',
                           help='Model without forecast (state_dim=14)')
    
    # Checkpoint selection
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename (default: best_model.pth)')
    
    # Display options
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed step information')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating evaluation plots')
    parser.add_argument('--no-stats', action='store_true',
                       help='Skip printing detailed statistics')
    
    return parser.parse_args()


class PolicyWrapper:
    """Simple wrapper for evaluation (actor only)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
    
    def load_actor(self, checkpoint_path):
        """Load trained actor from checkpoint"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.eval()
        print(f"✓ Loaded actor from: {checkpoint_path}")
        
        # Print checkpoint info if available
        if 'total_steps' in ckpt:
            print(f"  Training steps: {ckpt['total_steps']}")
        if 'epsilon' in ckpt:
            print(f"  Final epsilon: {ckpt['epsilon']:.4f}")
    
    def select_action(self, state):
        """Select action without noise (greedy policy)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        return action


def evaluate_agent(config, checkpoint_path, verbose=False):
    """
    Evaluate trained agent
    
    Args:
        config: Evaluation configuration
        checkpoint_path: Path to trained model checkpoint
        verbose: Print detailed step information
        
    Returns:
        eval_df: DataFrame with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"{'EVALUATION MODE':^70}")
    print(f"{'='*70}\n")
    
    # Initialize environment
    env = HVACEnvironment(config, verbose=verbose)
    
    # Initialize policy
    policy = PolicyWrapper(
        config.STATE_DIM, 
        config.ACTION_DIM,
        config.HIDDEN_DIM,
        device=config.DEVICE
    )
    policy.load_actor(checkpoint_path)
    
    # Reset environment
    state = env.reset()
    
    # Evaluation loop
    done = False
    step_count = 0
    cumulative_reward = 0.0
    
    log = {
        "time_hours": [],
        "T_zone_C": [],
        "RH_zone": [],
        "CO2_ppm": [],
        "P_total_kW": [],
        "reward": [],
        "uFan": [],
        "uOA": [],
        "uChiller": [],
        "uHeater": [],
        "uFanEA": [],
        "T_outdoor": [],
        "RH_outdoor": []
    }
    
    print("\n🚀 Running evaluation (no exploration noise)...\n")
    
    while not done:
        # Select action (greedy)
        action = policy.select_action(state)
        
        # Environment step
        next_state, reward, done, info = env.step(action)
        
        cumulative_reward += reward
        step_count += 1
        
        # Log data
        t_hours = (env.current_t - env.main_start) / 3600.0
        
        if info:
            log["time_hours"].append(t_hours)
            log["T_zone_C"].append(info["T_zone"])
            log["RH_zone"].append(info["RH_zone"])
            log["CO2_ppm"].append(info["CO2"])
            log["P_total_kW"].append(info["P_total"] / 1000.0)
            log["reward"].append(reward)
            log["uFan"].append(info["uFan"])
            log["uOA"].append(info["uOA"])
            log["uChiller"].append(info["uChiller"])
            log["uHeater"].append(info["uHeater"])
            log["uFanEA"].append(info["uFanEA"])
            log["T_outdoor"].append(info.get("T_outdoor", np.nan))
            log["RH_outdoor"].append(info.get("RH_outdoor", np.nan))
            
            # Progress update (if not verbose)
            if not verbose and step_count % 100 == 0:
                print(f"Step {step_count:04d} | t={t_hours:7.2f}h | "
                      f"T={info['T_zone']:5.2f}°C | RH={info['RH_zone']:5.3f} | "
                      f"CO2={info['CO2']:6.0f}ppm | P={info['P_total']/1000:5.2f}kW | "
                      f"r={reward:7.3f}")
        
        state = next_state
    
    # Summary statistics
    avg_reward = cumulative_reward / max(step_count, 1)
    print(f"\n{'='*70}")
    print(f"📋 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Steps:       {step_count}")
    print(f"  Total Reward:      {cumulative_reward:.3f}")
    print(f"  Avg Reward/Step:   {avg_reward:.5f}")
    print(f"{'='*70}\n")
    
    # Convert to DataFrame
    eval_df = pd.DataFrame(log)
    
    # Save CSV
    csv_path = config.EVAL_RESULTS_PATH / "eval_run_data.csv"
    eval_df.to_csv(csv_path, index=False)
    print(f"✓ Evaluation data saved to: {csv_path}")
    
    return eval_df


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine mode
    use_forecast = args.forecast
    mode_str = "WITH FORECAST" if use_forecast else "NO FORECAST"
    
    print(f"\n{'='*70}")
    print(f"HVAC DDPG EVALUATION - {mode_str}".center(70))
    print(f"{'='*70}\n")
    
    # Get configuration
    config = get_eval_config(use_forecast=use_forecast)
    config.create_directories()
    
    # Checkpoint path
    checkpoint_path = config.CHECKPOINT_PATH / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\n📂 Available checkpoints:")
        for ckpt in sorted(config.CHECKPOINT_PATH.glob("*.pth")):
            file_size = ckpt.stat().st_size / (1024 * 1024)  # MB
            print(f"   - {ckpt.name} ({file_size:.2f} MB)")
        sys.exit(1)
    
    # Evaluate agent
    eval_df = evaluate_agent(config, checkpoint_path, verbose=args.verbose)
    
    # Print comfort statistics (unless disabled)
    if not args.no_stats:
        print_comfort_statistics(eval_df)
    
    # Plot results (unless disabled)
    if not args.no_plot:
        try:
            plot_evaluation_results(eval_df, save_dir=str(config.EVAL_RESULTS_PATH))
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots: {e}")
    
    # Save summary report
    T_in_comfort = ((eval_df['T_zone_C'] >= 25.5) & (eval_df['T_zone_C'] <= 28.0)).sum()
    T_comfort_pct = (T_in_comfort / len(eval_df)) * 100
    
    RH_in_comfort = ((eval_df['RH_zone'] >= 0.40) & (eval_df['RH_zone'] <= 0.70)).sum()
    RH_comfort_pct = (RH_in_comfort / len(eval_df)) * 100
    
    both_comfort = ((eval_df['T_zone_C'] >= 25.5) & (eval_df['T_zone_C'] <= 28.0) &
                    (eval_df['RH_zone'] >= 0.40) & (eval_df['RH_zone'] <= 0.70)).sum()
    both_comfort_pct = (both_comfort / len(eval_df)) * 100
    
    summary = {
        'model_info': {
            'mode': mode_str,
            'state_dim': config.STATE_DIM,
            'checkpoint': args.checkpoint
        },
        'comfort': {
            'temperature_comfort_pct': float(T_comfort_pct),
            'humidity_comfort_pct': float(RH_comfort_pct),
            'combined_comfort_pct': float(both_comfort_pct)
        },
        'energy': {
            'mean_power_kW': float(eval_df['P_total_kW'].mean()),
            'total_energy_kWh': float(eval_df['P_total_kW'].sum() * 0.25)
        }
    }
    
    summary_path = config.EVAL_RESULTS_PATH / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary report saved to: {summary_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE".center(70))
    print(f"{'='*70}")
    print(f"Mode:                {mode_str}")
    print(f"Temperature Comfort: {T_comfort_pct:.1f}%")
    print(f"Humidity Comfort:    {RH_comfort_pct:.1f}%")
    print(f"Combined Comfort:    {both_comfort_pct:.1f}%")
    print(f"Total Energy:        {eval_df['P_total_kW'].sum() * 0.25:.2f} kWh")
    print(f"Results Directory:   {config.EVAL_RESULTS_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
