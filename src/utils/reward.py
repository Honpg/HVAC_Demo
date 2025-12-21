"""
Hierarchical Reward Calculator
"""
import numpy as np


class HierarchicalRewardCalculator:
    """Hierarchical reward: Temperature > Humidity > Energy"""
    
    def __init__(self, config, curriculum_enabled=True):
        # Load from config
        self.T_target_low = config.T_TARGET_LOW
        self.T_target_high = config.T_TARGET_HIGH
        self.T_init_low = config.T_INIT_LOW
        self.T_init_high = config.T_INIT_HIGH
        self.T_severe_low = config.T_SEVERE_LOW
        self.T_severe_high = config.T_SEVERE_HIGH
        
        self.RH_target_low = config.RH_TARGET_LOW
        self.RH_target_high = config.RH_TARGET_HIGH
        self.RH_init_low = config.RH_INIT_LOW
        self.RH_init_high = config.RH_INIT_HIGH
        self.RH_severe_low = config.RH_SEVERE_LOW
        self.RH_severe_high = config.RH_SEVERE_HIGH
        
        self.wT = config.WEIGHT_TEMP
        self.wRH = config.WEIGHT_HUMIDITY
        self.wEnergy = config.WEIGHT_ENERGY
        self.wT_unocc = config.WEIGHT_TEMP_UNOCCUPIED
        self.wRH_unocc = config.WEIGHT_HUMIDITY_UNOCCUPIED
        self.wEnergy_unocc = config.WEIGHT_ENERGY_UNOCCUPIED
        
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_max_episodes = config.CURRICULUM_MAX_EPISODES
        self.curriculum_episode = 0
        
        # Current bands
        self.T_comf_low = self.T_init_low
        self.T_comf_high = self.T_init_high
        self.RH_comf_low = self.RH_init_low
        self.RH_comf_high = self.RH_init_high
        
        # Fixed parameters
        self.severe_penalty = 5.0
        self.comfort_bonus = 2.0
        self.stability_bonus = 0.5
        self.smoothness_weight = 0.15
        self.action_change_bonus = 0.3
        self.action_extreme_penalty = 0.2
        self.P_ref_kW = 2.5
        self.r_min = -10.0
        self.r_max = 5.0
        self.occupancy_threshold = 0.3
        
        # Tracking
        self.prev_T = None
        self.prev_action_stored = None
        self.comfort_steps = 0
        self.action_changes = 0
        
    def update_curriculum(self, episode):
        if not self.curriculum_enabled:
            self.T_comf_low = self.T_target_low
            self.T_comf_high = self.T_target_high
            self.RH_comf_low = self.RH_target_low
            self.RH_comf_high = self.RH_target_high
            return
            
        self.curriculum_episode = episode
        progress = min(1.0, episode / self.curriculum_max_episodes)
        
        self.T_comf_low = self.T_init_low + progress * (self.T_target_low - self.T_init_low)
        self.T_comf_high = self.T_init_high + progress * (self.T_target_high - self.T_init_high)
        self.RH_comf_low = self.RH_init_low + progress * (self.RH_target_low - self.RH_init_low)
        self.RH_comf_high = self.RH_init_high + progress * (self.RH_target_high - self.RH_init_high)
    
    def _compute_deviation(self, x, low, high, severe_low, severe_high):
        if low <= x <= high:
            return 0.0
        elif x < low:
            if x <= severe_low:
                norm_dist = (severe_low - x) / (severe_low - 20.0)
                return (1.0 + norm_dist) ** 3
            else:
                norm_dist = (low - x) / (low - severe_low)
                return norm_dist ** 2
        else:
            if x >= severe_high:
                norm_dist = (x - severe_high) / (40.0 - severe_high)
                return (1.0 + norm_dist) ** 3
            else:
                norm_dist = (x - high) / (severe_high - high)
                return norm_dist ** 2
    
    def _compute_action_penalties(self, action, prev_action):
        extreme_penalty = 0.0
        change_bonus = 0.0
        
        for a in action:
            if a < 0.05 or a > 0.95:
                extreme_penalty += self.action_extreme_penalty
        
        if prev_action is not None:
            action_diff = np.abs(action - prev_action)
            mean_change = np.mean(action_diff)
            
            if 0.05 < mean_change < 0.3:
                change_bonus = self.action_change_bonus
                self.action_changes += 1
            elif mean_change < 0.02:
                extreme_penalty += 0.1
        
        return extreme_penalty, change_bonus
    
    def compute_reward(self, T_zone_K, RH_zone_frac, CO2_zone_ppm, P_total_W,
                       action, prev_action, occupancy, num_people):
        T_C = T_zone_K - 273.15
        RH = RH_zone_frac
        P_kW = P_total_W / 1000.0
        
        dT = self._compute_deviation(T_C, self.T_comf_low, self.T_comf_high,
                                     self.T_severe_low, self.T_severe_high)
        dRH = self._compute_deviation(RH, self.RH_comf_low, self.RH_comf_high,
                                      self.RH_severe_low, self.RH_severe_high)
        
        E_norm = min(P_kW / self.P_ref_kW, 2.0)
        
        if occupancy >= self.occupancy_threshold:
            wT, wRH, wE = self.wT, self.wRH, self.wEnergy
        else:
            wT, wRH, wE = self.wT_unocc, self.wRH_unocc, self.wEnergy_unocc
        
        T_cost = wT * dT
        RH_cost = wRH * dRH if dT < 0.5 else wRH * 0.3 * dRH
        energy_cost = wE * (E_norm ** 2) if (dT < 0.5 and dRH < 0.5) else wE * 0.2 * (E_norm ** 2)
        
        action_extreme_penalty, action_change_bonus = self._compute_action_penalties(action, prev_action)
        
        smoothness_cost = 0.0
        if prev_action is not None:
            smoothness_cost = self.smoothness_weight * np.sum((action - prev_action) ** 2)
        
        severe_penalty = 0.0
        if T_C <= self.T_severe_low or T_C >= self.T_severe_high:
            severe_penalty += self.severe_penalty
        if RH <= self.RH_severe_low or RH >= self.RH_severe_high:
            severe_penalty += self.severe_penalty * 0.7
        
        combo_penalty = 0.0
        if T_C < self.T_comf_low and RH > self.RH_comf_high:
            combo_penalty += 2.0
        if T_C > self.T_comf_high and RH < self.RH_comf_low:
            combo_penalty += 1.5
        
        total_cost = (T_cost + RH_cost + energy_cost + smoothness_cost +
                     severe_penalty + combo_penalty + action_extreme_penalty)
        reward = 2.0 - total_cost
        
        if (self.T_comf_low <= T_C <= self.T_comf_high and
            self.RH_comf_low <= RH <= self.RH_comf_high and
            occupancy >= self.occupancy_threshold):
            reward += self.comfort_bonus
            self.comfort_steps += 1
            if self.comfort_steps > 10:
                reward += self.stability_bonus
        else:
            self.comfort_steps = 0
        
        reward += action_change_bonus
        
        if self.prev_T is not None:
            T_change = abs(T_C - self.prev_T)
            if T_change < 0.5 and self.T_comf_low <= T_C <= self.T_comf_high:
                reward += 0.3
        
        self.prev_T = T_C
        self.prev_action_stored = action.copy() if action is not None else None
        
        reward = max(self.r_min, min(self.r_max, reward))
        return reward
    
    def reset_episode(self, episode):
        self.update_curriculum(episode)
        self.comfort_steps = 0
        self.prev_T = None
        self.prev_action_stored = None
        self.action_changes = 0
    
    def get_current_bands(self):
        return {
            'T_low': self.T_comf_low,
            'T_high': self.T_comf_high,
            'RH_low': self.RH_comf_low,
            'RH_high': self.RH_comf_high
        }
