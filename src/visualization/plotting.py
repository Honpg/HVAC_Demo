"""
Visualization utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_training_progress(episode_rewards, episode_stats, save_dir):
    """Plot training progress"""
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = range(1, len(episode_rewards) + 1)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Episode rewards
    axes[0, 0].plot(episodes, episode_rewards, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature
    avg_temps = [s['avg_T'] for s in episode_stats]
    std_temps = [s['std_T'] for s in episode_stats]
    axes[0, 1].plot(episodes, avg_temps, 'r-', linewidth=2)
    axes[0, 1].fill_between(episodes, 
                            np.array(avg_temps) - np.array(std_temps),
                            np.array(avg_temps) + np.array(std_temps),
                            alpha=0.3)
    axes[0, 1].axhline(26.0, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(27.5, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Average Zone Temperature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Humidity
    avg_rh = [s['avg_RH'] for s in episode_stats]
    axes[1, 0].plot(episodes, avg_rh, 'g-', linewidth=2)
    axes[1, 0].axhline(0.45, color='green', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0.65, color='green', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Relative Humidity')
    axes[1, 0].set_title('Average Zone Humidity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comfort ratios
    T_comfort = [s['T_comfort_ratio'] for s in episode_stats]
    RH_comfort = [s['RH_comfort_ratio'] for s in episode_stats]
    axes[1, 1].plot(episodes, T_comfort, 'r-', label='Temperature', linewidth=2)
    axes[1, 1].plot(episodes, RH_comfort, 'b-', label='Humidity', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Comfort Ratio')
    axes[1, 1].set_title('Comfort Compliance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    # Action diversity
    action_div = [s['action_diversity'] for s in episode_stats]
    axes[2, 0].plot(episodes, action_div, 'm-', linewidth=2)
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Action Diversity')
    axes[2, 0].set_title('Action Diversity')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Moving average
    window = min(5, len(episode_rewards))
    if window > 1:
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        axes[2, 1].plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Raw')
        axes[2, 1].plot(episodes, moving_avg, 'b-', linewidth=2, label=f'{window}-ep MA')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Reward')
        axes[2, 1].set_title('Reward Moving Average')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_progress.png', dpi=200)
    plt.close()
    print(f"✓ Training progress saved to {save_dir}/training_progress.png")


def plot_evaluation_results(log_df, save_dir):
    """Plot detailed evaluation results (12 plots)"""
    os.makedirs(save_dir, exist_ok=True)
    
    time_h = log_df['time_hours'].values
    
    # 1. Temperature
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['T_zone_C'], linewidth=2.5, color='#E63946', label='T_zone')
    ax.axhspan(25.5, 28.0, alpha=0.25, color='green', label='Comfort')
    if 'T_outdoor' in log_df.columns:
        ax.plot(time_h, log_df['T_outdoor'], linewidth=2, color='gray', 
                alpha=0.6, linestyle='--', label='T_outdoor')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Temperature [°C]', fontsize=16, fontweight='bold')
    ax.set_title('Zone Temperature', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_zone_temperature.png', dpi=300)
    plt.close()
    
    # 2. Humidity
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['RH_zone'], linewidth=2.5, color='#06A77D', label='RH_zone')
    ax.axhspan(0.40, 0.70, alpha=0.25, color='green', label='Comfort')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Relative Humidity', fontsize=16, fontweight='bold')
    ax.set_title('Zone Humidity', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_relative_humidity.png', dpi=300)
    plt.close()
    
    # 3. CO2
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['CO2_ppm'], linewidth=2.5, color='#F77F00')
    ax.axhline(800, linestyle='--', color='green', linewidth=2.5, label='Target')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('CO2 [ppm]', fontsize=16, fontweight='bold')
    ax.set_title('CO2 Concentration', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_co2_concentration.png', dpi=300)
    plt.close()
    
    # 4. Power
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['P_total_kW'], linewidth=2.5, color='#9D4EDD')
    ax.fill_between(time_h, 0, log_df['P_total_kW'], alpha=0.4, color='#9D4EDD')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Power [kW]', fontsize=16, fontweight='bold')
    ax.set_title('Total Power', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_total_power.png', dpi=300)
    plt.close()
    
    # 5. Reward
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['reward'], linewidth=2, color='#2E86AB', alpha=0.8)
    ax.axhline(0, linestyle='--', color='black', alpha=0.5, linewidth=2)
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=16, fontweight='bold')
    ax.set_title('Step Reward', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_step_reward.png', dpi=300)
    plt.close()
    
    # 6. Cumulative Reward
    fig, ax = plt.subplots(figsize=(16, 10))
    cumulative = np.cumsum(log_df['reward'])
    ax.plot(time_h, cumulative, linewidth=3, color='#06A77D')
    ax.fill_between(time_h, 0, cumulative, alpha=0.3, color='#06A77D')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=16, fontweight='bold')
    ax.set_title('Cumulative Reward', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_cumulative_reward.png', dpi=300)
    plt.close()
    
    # 7-9: Control Actions (3 plots)
    # Fan & OA
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['uFan'], linewidth=2.5, label='uFan', color='#E63946')
    ax.plot(time_h, log_df['uOA'], linewidth=2.5, label='uOA', color='#F77F00')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Action [0-1]', fontsize=16, fontweight='bold')
    ax.set_title('Fan & OA Control', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_fan_oa_control.png', dpi=300)
    plt.close()
    
    # Chiller & Heater
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['uChiller'], linewidth=2.5, label='uChiller', color='#06A77D')
    ax.plot(time_h, log_df['uHeater'], linewidth=2.5, label='uHeater', color='#9D4EDD')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Action [0-1]', fontsize=16, fontweight='bold')
    ax.set_title('Chiller & Heater', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/08_chiller_heater_control.png', dpi=300)
    plt.close()
    
    # Exhaust Fan
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_h, log_df['uFanEA'], linewidth=2.5, color='#2E86AB')
    ax.set_xlabel('Time [hours]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Action [0-1]', fontsize=16, fontweight='bold')
    ax.set_title('Exhaust Fan', fontsize=20, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/09_exhaust_fan_control.png', dpi=300)
    plt.close()
    
    # 10-12: Distributions (3 plots)
    # Temperature Distribution
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(log_df['T_zone_C'].dropna(), bins=60, color='#E63946', alpha=0.7, edgecolor='black')
    ax.axvline(25.5, color='green', linestyle='--', linewidth=3, label='Comfort Low')
    ax.axvline(28.0, color='green', linestyle='--', linewidth=3, label='Comfort High')
    ax.axvline(log_df['T_zone_C'].mean(), color='red', linestyle='-', linewidth=3, 
              label=f'Mean ({log_df["T_zone_C"].mean():.2f}°C)')
    ax.set_xlabel('Temperature [°C]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title('Temperature Distribution', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/10_temperature_distribution.png', dpi=300)
    plt.close()
    
    # Humidity Distribution
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(log_df['RH_zone'].dropna(), bins=60, color='#06A77D', alpha=0.7, edgecolor='black')
    ax.axvline(0.40, color='green', linestyle='--', linewidth=3, label='Comfort Low')
    ax.axvline(0.70, color='green', linestyle='--', linewidth=3, label='Comfort High')
    ax.axvline(log_df['RH_zone'].mean(), color='red', linestyle='-', linewidth=3,
              label=f'Mean ({log_df["RH_zone"].mean():.3f})')
    ax.set_xlabel('Humidity', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title('Humidity Distribution', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/11_humidity_distribution.png', dpi=300)
    plt.close()
    
    # Power Distribution
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(log_df['P_total_kW'].dropna(), bins=60, color='#9D4EDD', alpha=0.7, edgecolor='black')
    ax.axvline(log_df['P_total_kW'].mean(), color='red', linestyle='-', linewidth=3,
              label=f'Mean ({log_df["P_total_kW"].mean():.2f} kW)')
    ax.axvline(log_df['P_total_kW'].median(), color='orange', linestyle='--', linewidth=3,
              label=f'Median ({log_df["P_total_kW"].median():.2f} kW)')
    ax.set_xlabel('Power [kW]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title('Power Distribution', fontsize=20, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/12_power_distribution.png', dpi=300)
    plt.close()
    
    print(f"\n✅ All 12 plots saved to: {save_dir}/")


def print_comfort_statistics(log_df):
    """Print detailed comfort statistics"""
    print("\n" + "="*70)
    print("📊 COMFORT STATISTICS")
    print("="*70)
    
    T_in_comfort = ((log_df['T_zone_C'] >= 25.5) & (log_df['T_zone_C'] <= 28.0)).sum()
    T_comfort_pct = (T_in_comfort / len(log_df)) * 100
    
    RH_in_comfort = ((log_df['RH_zone'] >= 0.40) & (log_df['RH_zone'] <= 0.70)).sum()
    RH_comfort_pct = (RH_in_comfort / len(log_df)) * 100
    
    both_comfort = ((log_df['T_zone_C'] >= 25.5) & (log_df['T_zone_C'] <= 28.0) &
                    (log_df['RH_zone'] >= 0.40) & (log_df['RH_zone'] <= 0.70)).sum()
    both_comfort_pct = (both_comfort / len(log_df)) * 100
    
    print(f"\n🌡️  TEMPERATURE:")
    print(f"   Range:       25.5 - 28.0°C")
    print(f"   Mean:        {log_df['T_zone_C'].mean():.2f}°C")
    print(f"   Std:         {log_df['T_zone_C'].std():.2f}°C")
    print(f"   Min/Max:     {log_df['T_zone_C'].min():.2f} / {log_df['T_zone_C'].max():.2f}°C")
    print(f"   Comfort:     {T_comfort_pct:.1f}% ({T_in_comfort}/{len(log_df)})")
    
    print(f"\n💧 HUMIDITY:")
    print(f"   Range:       0.40 - 0.70")
    print(f"   Mean:        {log_df['RH_zone'].mean():.3f}")
    print(f"   Std:         {log_df['RH_zone'].std():.3f}")
    print(f"   Min/Max:     {log_df['RH_zone'].min():.3f} / {log_df['RH_zone'].max():.3f}")
    print(f"   Comfort:     {RH_comfort_pct:.1f}% ({RH_in_comfort}/{len(log_df)})")
    
    print(f"\n⚡ POWER:")
    print(f"   Mean:        {log_df['P_total_kW'].mean():.2f} kW")
    print(f"   Total:       {log_df['P_total_kW'].sum() * 0.25:.2f} kWh")
    
    print(f"\n✅ COMBINED COMFORT: {both_comfort_pct:.1f}%")
    print("="*70 + "\n")
