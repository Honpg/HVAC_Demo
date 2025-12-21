"""
HVAC FMU Environment with dual-mode support
"""
import numpy as np
import pandas as pd
from pyfmi import load_fmu

from ..utils import HierarchicalRewardCalculator


class HVACEnvironment:
    """HVAC Environment supporting both forecast and no-forecast modes"""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        
        # Load weather data
        self.weather_df = pd.read_csv(config.WEATHER_CSV)
        self.time_array = self.weather_df['time'].values
        self.t_min = float(self.time_array.min())
        self.t_max = float(self.time_array.max())
        self.dt = config.DT
        
        # Forecast handling
        self.forecast_col = 'T_forecast'
        if config.USE_FORECAST:
            if self.forecast_col not in self.weather_df.columns:
                print(f"⚠️  CSV missing '{self.forecast_col}'. Using TDryBul as fallback.")
                self.weather_df[self.forecast_col] = self.weather_df['TDryBul']
            self.forecast_array = self.weather_df[self.forecast_col].values
            print(f"✓ Running WITH forecast (state_dim={config.STATE_DIM})")
        else:
            self.forecast_array = None
            print(f"✓ Running WITHOUT forecast (state_dim={config.STATE_DIM})")
        
        # FMU setup
        self.fmu_path = config.FMU_PATH
        self.model = load_fmu(self.fmu_path)
        
        self.weather_input_names = ['TDryBul', 'relHum', 'pAtm', 'winSpe', 'HDirNor', 'HDifHor']
        self.control_input_names = ['uFan', 'uOA', 'uChiller', 'uHeater', 'occupancy', 'uFanEA']
        self.all_input_names = self.weather_input_names + self.control_input_names
        
        # Time configuration
        self.warmup_days = config.WARMUP_DAYS
        self.warmup_duration = self.warmup_days * 24 * 3600
        self.csv_duration = self.t_max - self.t_min
        self.main_start = self.warmup_duration
        self.main_end = self.main_start + self.csv_duration
        
        # State tracking
        self.current_t = None
        self.prev_fmu_outputs = None
        self.prev_weather = None
        self.prev_forecast = None
        self.prev_action = None
        self.occupancy = config.OCCUPANCY
        self.num_people = config.NUM_PEOPLE
        
        # Reward calculator
        self.reward_calculator = HierarchicalRewardCalculator(
            config=config,
            curriculum_enabled=config.CURRICULUM_ENABLED
        )
        
        self.current_episode = 0
        
        # Performance tracking
        self.step_rewards = []
        self.step_T = []
        self.step_RH = []
        self.step_actions = []
    
    def warmup(self):
        """7-day warmup phase"""
        if self.verbose:
            print(f"=== Warmup Phase ({self.warmup_days} days) ===")
        
        # Constants from first CSV row
        TDryBul_const = float(self.weather_df['TDryBul'].iloc[0])
        relHum_const = float(self.weather_df['relHum'].iloc[0])
        pAtm_const = float(self.weather_df['pAtm'].iloc[0])
        winSpe_const = float(self.weather_df['winSpe'].iloc[0])
        HDir_const = float(self.weather_df['HGloHor'].iloc[0])
        HDif_const = float(self.weather_df['HDifHor'].iloc[0])
        
        if self.config.USE_FORECAST:
            Tfore_const = float(self.weather_df[self.forecast_col].iloc[0])
        
        WARMUP_ACTION = [0.6, 0.5, 0.5, 0.3, 0.3, 0.5]
        
        n_warm_points = int(self.warmup_duration / self.dt) + 1
        time_warm_input = np.linspace(0.0, self.warmup_duration, n_warm_points)
        
        weather_warm = np.column_stack([
            TDryBul_const * np.ones(n_warm_points),
            relHum_const * np.ones(n_warm_points),
            pAtm_const * np.ones(n_warm_points),
            winSpe_const * np.ones(n_warm_points),
            HDir_const * np.ones(n_warm_points),
            HDif_const * np.ones(n_warm_points)
        ])
        
        control_warm = np.column_stack([
            np.ones(n_warm_points) * WARMUP_ACTION[0],
            np.ones(n_warm_points) * WARMUP_ACTION[1],
            np.ones(n_warm_points) * WARMUP_ACTION[2],
            np.ones(n_warm_points) * WARMUP_ACTION[3],
            np.ones(n_warm_points) * WARMUP_ACTION[4],
            np.ones(n_warm_points) * WARMUP_ACTION[5],
        ])
        
        input_warm_full = np.column_stack([time_warm_input, weather_warm, control_warm])
        input_object_warm = (self.all_input_names, input_warm_full)
        
        opts_warm = self.model.simulate_options()
        opts_warm['ncp'] = n_warm_points - 1
        opts_warm['CVode_options']['rtol'] = 1e-6
        opts_warm['CVode_options']['atol'] = 1e-6
        
        res_warm = self.model.simulate(
            start_time=0.0,
            final_time=self.warmup_duration,
            input=input_object_warm,
            options=opts_warm
        )
        
        if self.verbose:
            print(f"✓ Warmup completed at t = {self.warmup_duration/3600:.2f} hours")
        
        self.prev_fmu_outputs = [
            float(res_warm['T_zone'][-1]),
            float(res_warm['RH_zone'][-1]),
            float(res_warm['CO2_zone_ppm'][-1]),
            float(res_warm['T_SA'][-1]),
            float(res_warm['T_SA_afterCooling'][-1]),
            float(res_warm['RH_SA'][-1]),
            float(res_warm['Vdot_SA'][-1]),
            float(res_warm['P_total'][-1]),
        ]
        
        self.prev_weather = [TDryBul_const, relHum_const]
        
        if self.config.USE_FORECAST:
            self.prev_forecast = Tfore_const
        
        self.prev_action = np.array([
            WARMUP_ACTION[0], WARMUP_ACTION[1], WARMUP_ACTION[2],
            WARMUP_ACTION[3], WARMUP_ACTION[5]
        ])
        self.current_t = self.main_start
        
        return self.get_state()
    
    def get_state(self):
        """
        Build state vector based on mode:
        - WITH FORECAST (15D): [FMU(8), weather(2), forecast(1), time(2), occ(1), prev_act(1)]
        - NO FORECAST (14D):   [FMU(8), weather(2), time(2), occ(1), prev_act(1)]
        """
        hour_of_day = int(((self.current_t) // 3600) % 24)
        day_of_week = int(((self.current_t) // 86400) % 7)
        prev_action_avg = np.mean(self.prev_action) if self.prev_action is not None else 0.5
        
        base_state = self.prev_fmu_outputs + self.prev_weather
        
        if self.config.USE_FORECAST:
            forecast_val = (self.prev_forecast if self.prev_forecast is not None 
                          else self.weather_df[self.forecast_col].iloc[0])
            state = np.array(
                base_state +
                [forecast_val] +
                [hour_of_day / 24.0, day_of_week / 7.0] +
                [self.occupancy] +
                [prev_action_avg]
            )
        else:
            state = np.array(
                base_state +
                [hour_of_day / 24.0, day_of_week / 7.0] +
                [self.occupancy] +
                [prev_action_avg]
            )
        
        return state
    
    def step(self, action):
        """Execute one control step"""
        uFan = float(np.clip(action[0], 0.1, 0.8))
        uOA = float(np.clip(action[1], 0.3, 0.9))
        uChiller = float(np.clip(action[2], 0.0, 1.0))
        uHeater = float(np.clip(action[3], 0.0, 1.0))
        uFanEA = float(np.clip(action[4], 0.2, 0.7))
        
        self.model.set('uFan', uFan)
        self.model.set('uOA', uOA)
        self.model.set('uChiller', uChiller)
        self.model.set('uHeater', uHeater)
        self.model.set('occupancy', self.occupancy)
        self.model.set('uFanEA', uFanEA)
        
        t_vec = np.array([self.current_t, self.current_t + self.dt])
        t_weather = t_vec - self.warmup_duration + self.t_min
        
        Tdry_vec = np.interp(t_weather, self.time_array, self.weather_df['TDryBul'].values)
        relHum_vec = np.interp(t_weather, self.time_array, self.weather_df['relHum'].values)
        pAtm_vec = np.interp(t_weather, self.time_array, self.weather_df['pAtm'].values)
        winSpe_vec = np.interp(t_weather, self.time_array, self.weather_df['winSpe'].values)
        HDir_vec = np.interp(t_weather, self.time_array, self.weather_df['HGloHor'].values)
        HDif_vec = np.interp(t_weather, self.time_array, self.weather_df['HDifHor'].values)
        
        if self.config.USE_FORECAST:
            Tfor_vec = np.interp(t_weather, self.time_array, self.forecast_array)
        
        weather_step = np.column_stack([
            Tdry_vec, relHum_vec, pAtm_vec, winSpe_vec, HDir_vec, HDif_vec
        ])
        
        input_step_full = np.column_stack([t_vec, weather_step])
        input_object_step = (self.weather_input_names, input_step_full)
        
        opts_step = self.model.simulate_options()
        opts_step['initialize'] = False
        opts_step['ncp'] = 1
        opts_step['CVode_options']['rtol'] = 1e-6
        opts_step['CVode_options']['atol'] = 1e-6
        
        try:
            res = self.model.simulate(
                start_time=self.current_t,
                final_time=self.current_t + self.dt,
                input=input_object_step,
                options=opts_step
            )
            
            outputs_t = [
                float(res['T_zone'][-1]),
                float(res['RH_zone'][-1]),
                float(res['CO2_zone_ppm'][-1]),
                float(res['T_SA'][-1]),
                float(res['T_SA_afterCooling'][-1]),
                float(res['RH_SA'][-1]),
                float(res['Vdot_SA'][-1]),
                float(res['P_total'][-1]),
            ]
            
            reward = self.reward_calculator.compute_reward(
                T_zone_K=outputs_t[0],
                RH_zone_frac=outputs_t[1],
                CO2_zone_ppm=outputs_t[2],
                P_total_W=outputs_t[7],
                action=action,
                prev_action=self.prev_action,
                occupancy=self.occupancy,
                num_people=self.num_people
            )
            
            T_zone_C = outputs_t[0] - 273.15
            
            self.step_rewards.append(reward)
            self.step_T.append(T_zone_C)
            self.step_RH.append(outputs_t[1])
            self.step_actions.append(action.copy())
            
            info = {
                'T_zone': T_zone_C,
                'RH_zone': outputs_t[1],
                'CO2': outputs_t[2],
                'P_total': outputs_t[7],
                'uFan': uFan,
                'uOA': uOA,
                'uChiller': uChiller,
                'uHeater': uHeater,
                'uFanEA': uFanEA,
                'T_outdoor': Tdry_vec[-1] - 273.15,
                'RH_outdoor': relHum_vec[-1]
            }
            
            if self.config.USE_FORECAST:
                info['T_forecast'] = Tfor_vec[-1] - 273.15
            
            if self.verbose:
                bands = self.reward_calculator.get_current_bands()
                print(
                    f"[t={self.current_t/3600:.2f}h] "
                    f"T={T_zone_C:.2f}°C [{bands['T_low']:.1f}-{bands['T_high']:.1f}] "
                    f"RH={outputs_t[1]:.3f} [{bands['RH_low']:.2f}-{bands['RH_high']:.2f}] "
                    f"P={outputs_t[7]/1000:.2f}kW | r={reward:.3f}"
                )
            
            self.prev_fmu_outputs = outputs_t
            self.prev_weather = [Tdry_vec[-1], relHum_vec[-1]]
            
            if self.config.USE_FORECAST:
                self.prev_forecast = Tfor_vec[-1]
            
            self.prev_action = action
            self.current_t += self.dt
            
            done = self.current_t >= self.main_end
            next_state = self.get_state()
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"⛔ FMU error at t={self.current_t/3600:.2f}h: {str(e)[:200]}")
            return self.get_state(), -10.0, True, {}
    
    def reset(self, episode=None):
        """Reset for new episode"""
        if episode is not None:
            self.current_episode = episode
            self.reward_calculator.reset_episode(episode)
        
        self.model = load_fmu(self.fmu_path)
        
        self.step_rewards = []
        self.step_T = []
        self.step_RH = []
        self.step_actions = []
        
        return self.warmup()
    
    def get_episode_stats(self):
        """Get episode statistics"""
        if len(self.step_T) == 0:
            return {}
        
        bands = self.reward_calculator.get_current_bands()
        
        T_in_comfort = np.sum(
            (np.array(self.step_T) >= bands['T_low']) & 
            (np.array(self.step_T) <= bands['T_high'])
        ) / len(self.step_T)
        
        RH_in_comfort = np.sum(
            (np.array(self.step_RH) >= bands['RH_low']) & 
            (np.array(self.step_RH) <= bands['RH_high'])
        ) / len(self.step_RH)
        
        if len(self.step_actions) > 1:
            actions_array = np.array(self.step_actions)
            action_diversity = np.mean(np.std(actions_array, axis=0))
        else:
            action_diversity = 0.0
        
        return {
            'avg_reward': np.mean(self.step_rewards),
            'avg_T': np.mean(self.step_T),
            'std_T': np.std(self.step_T),
            'avg_RH': np.mean(self.step_RH),
            'std_RH': np.std(self.step_RH),
            'T_comfort_ratio': T_in_comfort,
            'RH_comfort_ratio': RH_in_comfort,
            'action_diversity': action_diversity,
            'action_changes': self.reward_calculator.action_changes,
            'T_band': f"{bands['T_low']:.1f}-{bands['T_high']:.1f}",
            'RH_band': f"{bands['RH_low']:.2f}-{bands['RH_high']:.2f}"
        }
