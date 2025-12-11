"""
HVAC + RL Backend Engine
========================
Real FMU + DDPG integration for HVAC control.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from datetime import datetime

from pyfmi import load_fmu
from backend.weather_api import convert_weather_to_csv_format, replace_weather_in_csv


# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File/folder layout mới dưới thư mục setup/
FMU_DIR = os.path.join(BASE_DIR, "setup", "env__FMU")
MODEL_DIR = os.path.join(BASE_DIR, "setup", "model")
WEATHER_DIR = os.path.join(BASE_DIR, "setup", "weartherdata")

FMU_PATH = os.path.join(FMU_DIR, "AHU_FMU_Core_WeatherInput.fmu")

# CSV base + realtime per region
REGION_CONFIG = {
    # Đà Nẵng - giữ nguyên
    "DN": {
        "model": os.path.join(MODEL_DIR, "DDPG_DN.pth"),
        "base_csv": os.path.join(WEATHER_DIR, "weather_data_DN.csv"),
        "realtime_csv": os.path.join(WEATHER_DIR, "weather_data_realtime_DN.csv"),
    },
    # Hà Nội (HN) dùng model hot + file Ha_Dong, realtime HN
    "HN": {
        "model": os.path.join(MODEL_DIR, "best_model_HN_hot.pth"),
        "base_csv": os.path.join(WEATHER_DIR, "Ha_Dong_weather_data_test_M07_M08.csv"),
        "realtime_csv": os.path.join(WEATHER_DIR, "weather_data_realtime_HN.csv"),
    },
    # Sài Gòn (SG) dùng model cold + file Nha Bè, realtime SG
    "SG": {
        "model": os.path.join(MODEL_DIR, "best_model_SG_cold.pth"),
        "base_csv": os.path.join(WEATHER_DIR, "Nha_Be_weather_data_cold_M11_M12.csv"),
        "realtime_csv": os.path.join(WEATHER_DIR, "weather_data_realtime_SG.csv"),
    },
}

os.makedirs(WEATHER_DIR, exist_ok=True)

def get_region_config(region: str) -> Dict[str, str]:
    key = region.upper() if isinstance(region, str) else "DN"
    return REGION_CONFIG.get(key, REGION_CONFIG["DN"])


# Action warmup:
# - Baseline (Initial) per-region để dễ tinh chỉnh (SG/HN giống DN mặc định)
# - Predict dùng bộ chung, có thể tách riêng sau nếu cần.
BASELINE_WARMUP_ACTIONS = {
    "DN": [0.45, 0.4, 0.6, 0.45, 0.3, 0.3],
    "HN": [0.45, 0.4, 0.62, 0.55, 0.3, 0.3],
    "SG": [0.45, 0.4, 0.61, 0.5, 0.3, 0.3],
}
PREDICT_WARMUP_ACTION = [0.6, 0.5, 0.5, 0.3, 0.3, 0.5]


def get_baseline_warmup_action(region: str) -> list:
    """Return region-specific warmup action for Initial baseline."""
    key = region.upper() if isinstance(region, str) else "DN"
    return BASELINE_WARMUP_ACTIONS.get(key, BASELINE_WARMUP_ACTIONS["DN"])


# ==================== DDPG NETWORKS ====================

class Actor(nn.Module):
    """
    Actor network kiến trúc giống file training `ddpg_ver_7.py`
    (4 fully-connected layers + LayerNorm), để load đúng `ddpg_best.pth`.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        action = torch.sigmoid(self.fc4(x))
        return action


class Critic(nn.Module):
    """Critic Network: (State, Action) → Q-value"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value


class DDPGAgent:
    """DDPG Agent for inference only"""
    def __init__(self, state_dim=14, action_dim=5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        
        # Actor network only (for inference)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor.eval()
        
    def select_action(self, state):
        """Select action using actor network (no noise for evaluation)"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        return action
    
    def load(self, filepath):
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        print(f"✓ Loaded DDPG model from: {filepath}")


# ==================== HVAC FMU ENVIRONMENT ====================

class HVACEnvironment:
    """HVAC FMU Environment for simulation"""
    def __init__(
        self,
        fmu_path: str,
        weather_csv_path: str,
        dt: float = 900.0,
        warmup_weather_path: Optional[str] = None,
    ):
        self.dt = dt
        self.fmu_path = fmu_path
        
    # Load weather data cho phần main (realtime CSV được build mỗi lần predict)
        self.weather_df = pd.read_csv(weather_csv_path)
        self.time_array = self.weather_df['time'].values
        self.t_min = float(self.time_array.min())
        self.t_max = float(self.time_array.max())

        # Warmup weather: ưu tiên file warmup truyền vào, fallback dùng chính weather_df
        self.warmup_weather_df = None
        if warmup_weather_path and os.path.exists(warmup_weather_path):
            self.warmup_weather_df = pd.read_csv(warmup_weather_path)
        else:
            self.warmup_weather_df = self.weather_df.copy()
        
        # Load FMU
        self.model = load_fmu(fmu_path)
        
        # Input names
        self.weather_input_names = ['TDryBul', 'relHum', 'pAtm', 'winSpe', 'HDirNor', 'HDifHor']
        self.control_input_names = ['uFan', 'uOA', 'uChiller', 'uHeater', 'occupancy', 'uFanEA']
        self.all_input_names = self.weather_input_names + self.control_input_names
        
        # Warmup config: 7 days
        self.warmup_hours = 7 * 24  # 7 days
        self.warmup_duration = self.warmup_hours * 3600  # seconds
        self.csv_duration = self.t_max - self.t_min
        self.main_start = self.warmup_duration
        self.main_end = self.main_start + self.csv_duration
        
        # State tracking
        self.current_t = None
        self.prev_fmu_outputs = None
        self.prev_weather = None
        self.prev_action = None
        self.occupancy = 0.3
        self.num_people = 10
        
    def warmup(self):
        """Run warmup phase"""
        print(f"=== Starting Warmup Phase ({self.warmup_hours} hours) ===")
        
        warm_df = self.warmup_weather_df if self.warmup_weather_df is not None else self.weather_df
        TDryBul_const = float(warm_df['TDryBul'].iloc[0])
        relHum_const = float(warm_df['relHum'].iloc[0])
        pAtm_const = float(warm_df['pAtm'].iloc[0])
        winSpe_const = float(warm_df['winSpe'].iloc[0])
        HDir_const = float(warm_df['HGloHor'].iloc[0])
        HDif_const = float(warm_df['HDifHor'].iloc[0])
        
        # Warmup action (uFan, uOA, uChiller, uHeater, occupancy, uFanEA)
        # Cho phép override qua thuộc tính warmup_action_override (ví dụ cho Initial)
        if hasattr(self, "warmup_action_override") and self.warmup_action_override is not None:
            WARMUP_ACTION = self.warmup_action_override
        else:
            # Mặc định dùng action warmup cho luồng predict
            WARMUP_ACTION = PREDICT_WARMUP_ACTION
        
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
        
        print(f"✓ Warmup completed")
        
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
        # prev_action là vector 5 phần tử (không gồm occupancy)
        self.prev_action = np.array([
            WARMUP_ACTION[0],
            WARMUP_ACTION[1],
            WARMUP_ACTION[2],
            WARMUP_ACTION[3],
            WARMUP_ACTION[5],
        ])
        # Lưu lại trạng thái cuối warmup để hiển thị Initial IAQ/Action
        self.warmup_final_outputs = list(self.prev_fmu_outputs)
        self.warmup_action = self.prev_action.copy()
        self.current_t = self.main_start
        
        return self.get_state()
    
    def get_state(self):
        """Build 14-dimensional state vector"""
        hour_of_day = int(((self.current_t) // 3600) % 24)
        day_of_week = int(((self.current_t) // 86400) % 7)
        prev_action_avg = np.mean(self.prev_action) if self.prev_action is not None else 0.5
        
        state = np.array(
            self.prev_fmu_outputs +
            self.prev_weather +
            [hour_of_day / 24.0, day_of_week / 7.0] +
            [self.occupancy] +
            [prev_action_avg]
        )
        return state
    
    def step(self, action):
        """Execute one simulation step"""
        # Mapping action → control theo đúng phạm vi lúc training (ddpg_ver_7.py)
        uFan = float(np.clip(action[0], 0.1, 0.9))
        uOA = float(np.clip(action[1], 0.3, 1.0))
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
            
            self.prev_fmu_outputs = outputs_t
            self.prev_weather = [Tdry_vec[-1], relHum_vec[-1]]
            self.prev_action = action
            self.current_t += self.dt
            
            done = self.current_t >= self.main_end
            
            return self.get_state(), done, {
                'T_zone': outputs_t[0] - 273.15,
                'RH_zone': outputs_t[1],
                'CO2': outputs_t[2],
                'P_total': outputs_t[7] / 1000.0,
                'uFan': uFan,
                'uOA': uOA,
                'uChiller': uChiller,
                'uHeater': uHeater,
                'uFanEA': uFanEA
            }
            
        except Exception as e:
            print(f"⛔ FMU error: {str(e)[:100]}")
            raise Exception(f"FMU simulation failed: {str(e)}")  # Không fallback, raise error
    
    def reset(self):
        """Reset environment"""
        self.model = load_fmu(self.fmu_path)
        return self.warmup()


# ==================== REAL-TIME PREDICTION ====================

def predict_single_point(weather_data: Dict[str, Any], region: str = "DN") -> Dict[str, Any]:
    """
    Predict chỉ 1 điểm tại thời điểm real-time hiện tại.
    
    Args:
        weather_data: Dictionary từ weather_api.get_realtime_weather()
        
    Returns:
        Dictionary với prediction results:
        {
            "action": [uFan, uOA, uChiller, uHeater, uFanEA],
            "fmu_state": [14-dimensional state],
            "weather": weather_data,
            "processing": {...}
        }
    """
    start_time = time.time()
    
    cfg = get_region_config(region)
    realtime_csv = cfg["realtime_csv"]
    base_csv_path = cfg["base_csv"]
    model_path = cfg["model"]

    # Extract rounded hour (đã được làm tròn trong get_realtime_weather)
    rounded_hour = weather_data.get("rounded_hour", weather_data.get("current_hour", 0))
    
    # Build realtime CSV: append ngày cuối từ weather_data1.csv và replace trong ngày append
    replace_weather_in_csv(
        realtime_csv,
        weather_data,
        rounded_hour,
        base_csv_path=base_csv_path  # Base theo vùng
    )
    
    try:
        # PHASE 1: Lấy Initial IAQ/Action tại rounded_hour từ get_initial_baseline()
        baseline = get_initial_baseline(target_hour=rounded_hour, region=region)
        initial_iaq: Dict[str, float] = baseline.get("iaq", {})
        initial_action: Dict[str, float] = baseline.get("action_init", {})
        
        # PHASE 2: Predict realtime với file weather_data_realtime.csv (đã replace, 1 ngày)
        env = HVACEnvironment(
            fmu_path=FMU_PATH,
            weather_csv_path=realtime_csv,
            warmup_weather_path=base_csv_path,
            dt=900.0  # 15 minutes
        )
        # Bỏ warmup ở pha realtime: set warmup_duration = 0, nhưng vẫn dùng action warmup chuẩn để thống nhất
        env.warmup_hours = 0
        env.warmup_duration = 0
        env.main_start = 0
        env.main_end = env.t_max - env.t_min
        # Realtime không warmup, giữ action mặc định PREDICT_WARMUP_ACTION (không override)
        
        agent = DDPGAgent(state_dim=14, action_dim=5, device='cuda')
        agent.load(model_path)
        
        # Reset với warmup_duration=0 (không warmup)
        state = env.reset()
        
        # Tính thời điểm bắt đầu ngày append
        # Đọc lại CSV để tìm chính xác time min của ngày append
        # Ngày append là phần cuối của CSV (sau khi append từ base CSV)
        df_check = pd.read_csv(realtime_csv)
        # Tìm base CSV duration từ warmup_weather_df (base file theo vùng)
        if env.warmup_weather_df is not None:
            base_csv_max_time = float(env.warmup_weather_df['time'].max())
            # Ngày append bắt đầu từ base_max + 24h (hoặc + dt tùy cách append)
            # Trong replace_weather_in_csv, ta offset bằng day_offset = 24*3600
            # Nhưng cần tính chính xác từ CSV đã append
            # Tìm time min của các dòng có time > base_csv_max_time
            appended_times = df_check[df_check['time'] > base_csv_max_time + 1e-6]['time']
            if len(appended_times) > 0:
                appended_start_time = float(appended_times.min())
            else:
                # Fallback: tính từ base max + 24h
                appended_start_time = base_csv_max_time + 24 * 3600
        else:
            # Fallback: giả sử ngày append là nửa cuối của CSV
            mid_idx = len(df_check) // 2
            appended_start_time = float(df_check.iloc[mid_idx:]['time'].min())
        
        # Target time trong ngày append
        target_time_in_csv = appended_start_time + rounded_hour * 3600
        
        # Set current_t về đầu ngày append
        env.current_t = appended_start_time
        
        # Set initial state từ baseline để bắt đầu ngày append (đã lấy ở PHASE 1)
        initial_iaq_from_baseline = initial_iaq
        initial_action_from_baseline = initial_action
        
        # Convert initial IAQ to FMU output format
        env.prev_fmu_outputs = [
            initial_iaq_from_baseline.get("T_zone", 25.0) + 273.15,  # C to K
            initial_iaq_from_baseline.get("RH_zone", 50.0) / 100.0,  # % to ratio
            initial_iaq_from_baseline.get("CO2_zone", 500.0),
            0.0,  # T_SA (placeholder)
            0.0,  # T_SA_afterCooling (placeholder)
            0.0,  # RH_SA (placeholder)
            0.0,  # Vdot_SA (placeholder)
            initial_iaq_from_baseline.get("P_total", 0.0) * 1000.0  # kW to W
        ]
        
        # Set initial weather từ dòng đầu ngày append (defensive nếu filter rỗng)
        appended_mask = env.weather_df['time'] >= appended_start_time
        if not appended_mask.any():
            # Fallback: dùng dòng đầu tiên của CSV
            first_entry_appended = env.weather_df.iloc[0]
            appended_start_time = float(first_entry_appended['time'])
        else:
            first_entry_appended = env.weather_df.loc[appended_mask].iloc[0]
        env.prev_weather = [float(first_entry_appended['TDryBul']), float(first_entry_appended['relHum'])]
        
        # Set initial action từ baseline
        env.prev_action = np.array([
            initial_action_from_baseline.get("uFan", 0.5),
            initial_action_from_baseline.get("uOA", 0.5),
            initial_action_from_baseline.get("uChiller", 0.5),
            initial_action_from_baseline.get("uHeater", 0.0),
            initial_action_from_baseline.get("uFanEA", 0.5),
        ])
        
        # Get initial state cho ngày append
        state = env.get_state()
        
        # Tính số steps từ đầu ngày append đến target
        steps_needed = int((target_time_in_csv - appended_start_time) / env.dt)
        
        # Step đến target bằng RL agent
        for _ in range(max(0, steps_needed - 1)):
            action_mid = agent.select_action(state)
            state, done, _ = env.step(action_mid)
            if done:
                break
        
        # Predict action tại target và thực thi FMU
        # Giữ lại state/fmu_outputs trước khi áp dụng action để phản ánh thời điểm dự đoán
        state_before_action = state.copy()
        fmu_outputs_before = list(env.prev_fmu_outputs) if env.prev_fmu_outputs is not None else []
        
        action = agent.select_action(state_before_action)
        state_after, done, _ = env.step(action)
        
        # State trả về dùng BEFORE-action (tránh lệch +15 phút)
        current_state = state_before_action
        process_time_ms = (time.time() - start_time) * 1000
        
        # Ưu tiên fmu_outputs trước action; fallback sau action nếu thiếu
        fmu_outputs = fmu_outputs_before if fmu_outputs_before else env.prev_fmu_outputs
        
        result = {
            "action": action.tolist(),
            "action_names": ["uFan", "uOA", "uChiller", "uHeater", "uFanEA"],
            "action_dict": {
                "uFan": float(action[0]),
                "uOA": float(action[1]),
                "uChiller": float(action[2]),
                "uHeater": float(action[3]),
                "uFanEA": float(action[4])
            },
            "fmu_state": {
                "dimension": 14,
                "values": current_state.tolist(),
                "state_names": [
                    "T_zone", "RH_zone", "CO2_zone", "T_SA", "T_SA_afterCooling",
                    "RH_SA", "Vdot_SA", "P_total", "T_out", "relHum",
                    "hour_of_day", "day_of_week", "occupancy", "prev_action_avg"
                ],
                "fmu_outputs": {
                    "T_zone_K": float(fmu_outputs[0]),
                    "RH_zone_ratio": float(fmu_outputs[1]),
                    "RH_zone_percent": float(fmu_outputs[1]) * 100.0,
                    "CO2_zone_ppm": float(fmu_outputs[2]),
                    "P_total_W": float(fmu_outputs[7])
                }
            },
            "weather": weather_data,
            "initial": {
                "iaq": initial_iaq,
                "action_init": initial_action
            },
            "processing": {
                "warmup_hours": env.warmup_hours,
                "process_time_ms": round(process_time_ms, 2),
                "timestamp_generated": datetime.now().isoformat()
            }
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def get_initial_baseline(target_hour: int = 0, region: str = "DN") -> Dict[str, Dict[str, float]]:
    """
    Lấy Initial IAQ/Action thực từ FMU với weather_data1.csv tại giờ cụ thể.
    Chạy warmup 7 ngày với bộ action riêng cho Initial: [0.45, 0.4, 0.6, 0.45, 0.3, 0.3]
    Sau đó step đến target_hour trong ngày đầu tiên sau warmup và lấy IAQ/Action tại đó.
    
    Args:
        target_hour: Giờ cần lấy Initial IAQ/Action (0-23), mặc định 0
    """
    try:
        cfg = get_region_config(region)
        base_csv = cfg["base_csv"]

        init_env = HVACEnvironment(
            fmu_path=FMU_PATH,
            weather_csv_path=base_csv,
            warmup_weather_path=base_csv,
            dt=900.0
        )
        # Warmup baseline dùng bộ action riêng
        init_env.warmup_action_override = get_baseline_warmup_action(region)
        # Warmup 7 ngày cho luồng Initial với bộ action riêng
        init_env.warmup_hours = 7 * 24
        init_env.warmup_duration = init_env.warmup_hours * 3600.0
        init_env.main_start = init_env.warmup_duration
        init_env.main_end = init_env.main_start + init_env.csv_duration
        init_env.current_t = 0.0

        # Override warmup action cho Initial (thêm occupancy và uFanEA cuối)
        init_env.warmup_action_override = get_baseline_warmup_action(region)

        # Warmup
        state = init_env.reset()
        
        # Target time trong ngày đầu tiên sau warmup
        target_time = init_env.main_start + target_hour * 3600
        
        # Step từ main_start đến target_hour bằng action mặc định (không RL)
        default_action = np.array([0.45, 0.4, 0.6, 0.35, 0.3])  # Action mặc định cho Initial
        steps_needed = int((target_time - init_env.current_t) / init_env.dt)
        
        for _ in range(max(0, steps_needed)):
            state, done, _ = init_env.step(default_action)
            if done:
                break
        
        # Lấy IAQ/Action tại target_hour
        fmu_out = init_env.prev_fmu_outputs
        initial_iaq = {}
        if fmu_out:
            initial_iaq = {
                "T_zone": float(fmu_out[0]) - 273.15,
                "RH_zone": float(fmu_out[1]) * 100.0,
                "CO2_zone": float(fmu_out[2]),
                "P_total": float(fmu_out[7]) / 1000.0
            }

        # Action tại target_hour là action mặc định đã dùng
        initial_action = {
            "uFan": float(default_action[0]),
            "uOA": float(default_action[1]),
            "uChiller": float(default_action[2]),
            "uHeater": float(default_action[3]),
            "uFanEA": float(default_action[4]),
        }
        return {"iaq": initial_iaq, "action_init": initial_action}
    except Exception as e:
        print(f"⚠️ get_initial_baseline failed: {e}")
        return {"iaq": {}, "action_init": {}}
