"""
Base Configuration for HVAC DDPG Control
Unified config supporting both forecast and no-forecast modes
"""
import torch
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


class BaseConfig:
    """Base configuration class for HVAC DDPG"""
    
    # ==================== MODE SELECTION ====================
    USE_FORECAST = True  # Set False for state_dim=14, True for state_dim=15
    
    # ==================== GENERAL ====================
    PROJECT_NAME = "HVAC-DDPG-Control"
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================== STATE & ACTION ====================
    # State dimension automatically set based on USE_FORECAST
    @property
    def STATE_DIM(self):
        return 15 if self.USE_FORECAST else 14
    
    ACTION_DIM = 5  # [uFan, uOA, uChiller, uHeater, uFanEA]
    
    # ==================== NETWORK ====================
    HIDDEN_DIM = 512
    
    # ==================== TRAINING HYPERPARAMETERS ====================
    LR_ACTOR = 1e-4
    LR_CRITIC = 3e-4
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 512
    
    # ==================== REPLAY BUFFER ====================
    BUFFER_CAPACITY = 200000
    PRIORITIZED_REPLAY_ALPHA = 0.6
    PRIORITIZED_REPLAY_BETA = 0.4
    
    # ==================== EXPLORATION ====================
    EXPLORATION_STEPS = 200000
    EPSILON_START = 1.0
    EPSILON_END = 0.2
    EPSILON_DECAY = 0.9997
    
    # ==================== OU NOISE ====================
    OU_THETA = 0.15
    OU_SIGMA_START = 0.7
    OU_SIGMA_END = 0.15
    
    # ==================== LEARNING RATE SCHEDULER ====================
    LR_PATIENCE = 3
    LR_FACTOR = 0.5
    LR_MIN = 1e-7
    
    # ==================== ENVIRONMENT ====================
    FMU_PATH = str(DATA_DIR / "HVAC.fmu")
    WEATHER_CSV = str(DATA_DIR / "weather_data.csv")
    DT = 900.0  # 15 minutes
    WARMUP_DAYS = 7
    OCCUPANCY = 0.3
    NUM_PEOPLE = 10
    
    # ==================== TRAINING ====================
    NUM_EPISODES = 50
    SAVE_FREQ = 2
    
    # ==================== REWARD PARAMETERS ====================
    T_TARGET_LOW = 26.0
    T_TARGET_HIGH = 27.5
    T_INIT_LOW = 24.0
    T_INIT_HIGH = 29.0
    T_SEVERE_LOW = 24.0
    T_SEVERE_HIGH = 29.0
    
    RH_TARGET_LOW = 0.45
    RH_TARGET_HIGH = 0.65
    RH_INIT_LOW = 0.35
    RH_INIT_HIGH = 0.75
    RH_SEVERE_LOW = 0.35
    RH_SEVERE_HIGH = 0.75
    
    WEIGHT_TEMP = 5.0
    WEIGHT_HUMIDITY = 2.5
    WEIGHT_ENERGY = 4.0
    WEIGHT_TEMP_UNOCCUPIED = 1.5
    WEIGHT_HUMIDITY_UNOCCUPIED = 0.5
    WEIGHT_ENERGY_UNOCCUPIED = 6.0
    
    CURRICULUM_ENABLED = True
    CURRICULUM_MAX_EPISODES = 30
    
    # ==================== PATHS ====================
    @property
    def CHECKPOINT_PATH(self):
        mode = "with_forecast" if self.USE_FORECAST else "no_forecast"
        return CHECKPOINT_DIR / mode
    
    @property
    def RESULTS_PATH(self):
        mode = "with_forecast" if self.USE_FORECAST else "no_forecast"
        return RESULTS_DIR / "training" / mode
    
    @property
    def EVAL_RESULTS_PATH(self):
        mode = "with_forecast" if self.USE_FORECAST else "no_forecast"
        return RESULTS_DIR / "evaluation" / mode
    
    # ==================== METHODS ====================
    def create_directories(self):
        """Create necessary directories"""
        self.CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
        self.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        self.EVAL_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    def display(self):
        """Display configuration"""
        mode_str = "WITH FORECAST" if self.USE_FORECAST else "NO FORECAST"
        print("=" * 70)
        print(f"Configuration - {mode_str}".center(70))
        print("=" * 70)
        print(f"Device:           {self.DEVICE}")
        print(f"State Dim:        {self.STATE_DIM} ({'with forecast' if self.USE_FORECAST else 'no forecast'})")
        print(f"Action Dim:       {self.ACTION_DIM}")
        print(f"Hidden Dim:       {self.HIDDEN_DIM}")
        print(f"Actor LR:         {self.LR_ACTOR:.2e}")
        print(f"Critic LR:        {self.LR_CRITIC:.2e}")
        print(f"Batch Size:       {self.BATCH_SIZE}")
        print(f"Buffer Capacity:  {self.BUFFER_CAPACITY}")
        print(f"Episodes:         {self.NUM_EPISODES}")
        print(f"Save Frequency:   {self.SAVE_FREQ}")
        print(f"Checkpoint Path:  {self.CHECKPOINT_PATH}")
        print("=" * 70)
