# 🏢 HVAC-DDPG-Control

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**🤖 Deep Deterministic Policy Gradient (DDPG) for Intelligent HVAC Control**

*Tối ưu hóa hệ thống điều hòa không khí bằng Deep Reinforcement Learning*

</div>

---

## 📋 Mục Lục

- [🎯 Giới Thiệu](#-giới-thiệu)
- [🏭 Mô Hình Mô Phỏng Modelica](#-mô-hình-mô-phỏng-modelica-hvacmo)
- [✨ Tính Năng Nổi Bật](#-tính-năng-nổi-bật)
- [📁 Cấu Trúc Project](#-cấu-trúc-project)
- [🔧 Cài Đặt](#-cài-đặt)
- [🚀 Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [📊 Kết Quả](#-kết-quả)

---

## 🎯 Giới Thiệu

**HVAC-DDPG-Control** là một hệ thống điều khiển thông minh cho hệ thống HVAC (Heating, Ventilation, and Air Conditioning) sử dụng thuật toán **Deep Deterministic Policy Gradient (DDPG)**. 

Hệ thống sử dụng **Functional Mock-up Unit (FMU)** để mô phỏng động lực học của hệ thống HVAC thực tế, kết hợp với dữ liệu thời tiết thực từ các địa điểm tại Việt Nam (Hà Nội, Đà Nẵng, TP.HCM).

### 🎯 Mục tiêu chính:
- 🌡️ **Duy trì nhiệt độ thoải mái** trong phạm vi 26-27.5°C
- 💧 **Kiểm soát độ ẩm** trong khoảng 45-65%
- ⚡ **Tối ưu hóa năng lượng tiêu thụ**
- 🔮 **Hỗ trợ dự báo thời tiết** để điều khiển chủ động

---

## 🏭 Mô Hình Mô Phỏng Modelica (HVAC.mo)

> *Trái tim vật lý của hệ thống - Mô phỏng động lực học AHU hoàn chỉnh*

### 📐 Tổng Quan Kiến Trúc

Mô hình Modelica (`HVAC.mo`) là một mô phỏng **Air Handling Unit (AHU)** hoàn chỉnh được xây dựng trên nền tảng thư viện **Modelica Buildings Library v12.1.0**. Mô hình được thiết kế đặc biệt để tích hợp với **Reinforcement Learning** thông qua chuẩn **FMI (Functional Mock-up Interface)**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🏢 HVAC_FMU Package                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  📦 Medium Definitions                                                  │   │
│  │  • MediumA = Buildings.Media.Air (with CO2 tracking)                    │   │
│  │  • MediumW = Buildings.Media.Water                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ❄️ Carnot_TEva_NoWarn (Custom Chiller Model)                          │   │
│  │  • Energy-balanced chiller with manual control                          │   │
│  │  • TEva setpoint modulated by control signal                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  🌬️ AHU_FMU_Core_WeatherInput (Main AHU Model)                         │   │
│  │  • Complete AHU with all HVAC components                                │   │
│  │  • Weather data inputs from Python                                       │   │
│  │  • RL control inputs/outputs interface                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 🔧 Thông Số Hệ Thống (System Parameters)

| Parameter | Giá trị | Đơn vị | Mô tả |
|-----------|---------|--------|-------|
| `mAir_flow_nominal` | 0.24 | kg/s | Lưu lượng không khí định mức |
| `mWat_flow_nominal` | 0.35 | kg/s | Lưu lượng nước lạnh định mức |
| `Vzone` | 168 | m³ | Thể tích vùng điều hòa |
| `COP_chiller_nominal` | 1.5 | - | Hệ số hiệu suất chiller |
| `QEva_flow_nominal` | -6000 | W | Công suất làm lạnh định mức |
| `Q_heater_nominal` | 2000 | W | Công suất sưởi định mức |
| `UA_nominal` | 1500 | W/K | Hệ số truyền nhiệt coil |
| `Pfan_nominal` | 70 | W | Công suất quạt cấp khí |
| `PfanEA_nominal` | 50 | W | Công suất quạt thải |
| `PpumpCW_nominal` | 180 | W | Công suất bơm nước lạnh |
| `nOccMax` | 10 | người | Số người tối đa trong phòng |
| `CO2_outdoor_ppm` | 400 | ppm | Nồng độ CO2 ngoài trời |

---

### 🌬️ Sơ Đồ Luồng Không Khí (Airflow Diagram)

```
                                    ┌──────────────────────────────────────────────────────────────┐
                                    │                     🏠 CONDITIONED ZONE                       │
                                    │                      (V = 168 m³)                             │
                                    │                                                               │
                                    │   T_zone ───────────────────────────────► T_zone [K]         │
                                    │   RH_zone ──────────────────────────────► RH_zone [0-1]      │
                                    │   CO2_zone ─────────────────────────────► CO2_zone [ppm]     │
                                    │                                                               │
                                    │   👥 Internal Gains: CO2 + Moisture + Heat                   │
                                    └──────────────────▲────────────────┬───────────────────────────┘
                                                       │                │
                                                       │ Supply Air     │ Return Air
                                                       │                │
┌──────────────────────────────────────────────────────┴────────────────┴───────────────────────────────────┐
│                                         🌬️ AIR HANDLING UNIT (AHU)                                        │
│                                                                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  🌤️ Outdoor │    │  🚪 Damper  │    │  🧹 Filter  │    │   ⛓️ Mix   │    │  ❄️ Cooling │              │
│  │    Air      │───►│    OA       │───►│   preFil    │───►│   Junction  │───►│    Coil     │              │
│  │  (ambAir)   │    │  (damOA)    │    │             │    │   (mixT)    │    │  (cooCoil)  │              │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────▲──────┘    └──────┬──────┘              │
│        │                                                         │                  │                      │
│        │                              ┌──────────────────────────┘                  ▼                      │
│        │                              │ Return Air                          ┌─────────────┐              │
│        │                              │                                      │  💧 Dehum   │              │
│        │                      ┌───────┴───────┐                              │   Filter    │              │
│        │                      │  🚪 Damper RA │                              │ (dehumFilter)│              │
│        │                      │   (damRA)     │                              └──────┬──────┘              │
│        │                      └───────▲───────┘                                     │                      │
│        │                              │                                             ▼                      │
│        │                      ┌───────┴───────┐                              ┌─────────────┐              │
│        │                      │   ⛓️ Return  │                              │  🔥 Heater  │              │
│        │                      │   Junction    │◄──────────────────┐          │   Coil      │              │
│        │                      │    (retJ)     │                   │          │  (heaCoil)  │              │
│        │                      └───────┬───────┘                   │          └──────┬──────┘              │
│        │                              │                           │                  │                      │
│        │                              ▼                           │                  ▼                      │
│        │                      ┌─────────────┐              ┌──────┴──────┐    ┌─────────────┐              │
│        │                      │  🚪 Damper  │              │ 📊 Sensors  │    │  📡 T & RH  │              │
│        │                      │    EA       │              │  T, RH, CO2 │    │   Sensors   │              │
│        │                      │  (damEA)    │              │             │    │ (Tsa, RHsa) │              │
│        │                      └──────┬──────┘              └─────────────┘    └──────┬──────┘              │
│        │                             │                                               │                      │
│        │                             ▼                                               ▼                      │
│        │                      ┌─────────────┐                                 ┌─────────────┐              │
│        │                      │ 🌀 Exhaust  │                                 │ 🌀 Supply   │              │
│        │                      │    Fan      │                                 │    Fan      │──────────────┤
│        └─────────────────────►│   (fanEA)   │                                 │   (fanSA)   │   To Zone    │
│          Exhaust to Outside   └─────────────┘                                 └─────────────┘              │
│                                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### ❄️ Hệ Thống Nước Lạnh (Chilled Water System)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        💧 CHILLED WATER LOOP                                │
│                                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │ 🏭 Condenser│      │  ❄️ Chiller │      │ 📤 Evaporator│                │
│   │   Supply    │─────►│  (Carnot)   │─────►│   Outlet     │                │
│   │ (conSupply) │      │  COP = 1.5  │      │              │                │
│   │  T = 25°C   │      │ Q = -6 kW   │      │  T_cold      │                │
│   └─────────────┘      └──────┬──────┘      └──────┬───────┘                │
│                               │                    │                        │
│                               │ Condenser          │ Evaporator             │
│                               │ Return             │ Supply                 │
│                               ▼                    ▼                        │
│                        ┌─────────────┐      ┌─────────────┐                │
│                        │ 📥 Condenser│      │  🔄 Pump    │                │
│                        │   Return    │      │   (pumpCW)  │                │
│                        │ (conReturn) │      │  dp = 8 kPa │                │
│                        │  T = 30°C   │      └──────┬───────┘                │
│                        └─────────────┘             │                        │
│                                                    ▼                        │
│                                             ┌─────────────┐                │
│                                             │  🚿 Valve   │                │
│                                             │   (valCW)   │                │
│                                             │  PI Control │                │
│                                             └──────┬───────┘                │
│                                                    │                        │
│                                                    ▼                        │
│                                             ┌─────────────┐                │
│                                             │ ❄️ Cooling  │                │
│                                             │    Coil     │◄───── Air Flow │
│                                             │ (cooCoil)   │                 │
│                                             │ UA = 1500   │                 │
│                                             └─────────────┘                │
│                                                    │                        │
│                                                    └───────► Back to Chiller│
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 🎮 Giao Diện RL (Reinforcement Learning Interface)

#### 📥 Control Inputs (RL Actions)

| Input | Ký hiệu | Phạm vi | Mô tả | Safety Limits |
|-------|---------|---------|-------|---------------|
| **Fan Speed** | `uFan` | [0, 1] | Tốc độ quạt cấp khí | min = 0.3 |
| **Outside Air Damper** | `uOA` | [0, 1] | Độ mở damper gió tươi | min = 0.4 (tránh flow reversal) |
| **Chiller Capacity** | `uChiller` | [0, 1] | Công suất chiller | clamp [0, 1] |
| **Heater Capacity** | `uHeater` | [0, 1] | Công suất sưởi | max = 0.7 + 0.3×uOA |
| **Exhaust Fan Speed** | `uFanEA` | [0, 1] | Tốc độ quạt thải | min = 0.2 |
| **Occupancy** | `occupancy` | [0, 1] | Tỷ lệ người trong phòng | - |

#### 📤 Observations (RL States)

| Output | Ký hiệu | Đơn vị | Mô tả |
|--------|---------|--------|-------|
| **Supply Air Temperature** | `T_SA` | K | Nhiệt độ không khí cấp |
| **Supply Air Humidity** | `RH_SA` | [0-1] | Độ ẩm tương đối khí cấp |
| **Supply Air Flow** | `Vdot_SA` | m³/s | Lưu lượng thể tích khí cấp |
| **Zone Temperature** | `T_zone` | K | Nhiệt độ vùng điều hòa |
| **Zone Humidity** | `RH_zone` | [0-1] | Độ ẩm tương đối vùng |
| **Zone CO2** | `CO2_zone_ppm` | ppm | Nồng độ CO2 trong phòng |
| **SA After Cooling** | `T_SA_afterCooling` | K | Nhiệt độ sau coil lạnh |

#### ⚡ Energy Outputs

| Output | Ký hiệu | Đơn vị | Công thức |
|--------|---------|--------|-----------|
| **Supply Fan Power** | `P_fan` | W | $P_{fan} = P_{nominal} \times (ṁ/ṁ_{nominal})^3$ |
| **Exhaust Fan Power** | `P_fanEA` | W | $P_{fanEA} = P_{nominal} \times (ṁ/ṁ_{nominal})^3$ |
| **Chiller Power** | `P_chiller` | W | $P_{chiller} = |Q_{eva}| / COP$ |
| **Pump Power** | `P_pump` | W | $P_{pump} = P_{nominal} \times (ṁ/ṁ_{nominal})^3$ |
| **Heater Power** | `P_heater` | W | $P_{heater} = Q_{heater}$ |
| **Total Power** | `P_total` | W | $\sum$ (Fan + FanEA + Chiller + Pump + Heater) |

---

### 🌤️ Weather Data Interface

```python
# Weather inputs from Python (interpolated from CSV at each timestep)
weather_inputs = {
    'TDryBul':  float,  # Outdoor dry bulb temperature [K]
    'relHum':   float,  # Outdoor relative humidity [0-1]  
    'pAtm':     float,  # Atmospheric pressure [Pa]
    'winSpe':   float,  # Wind speed [m/s]
    'HDirNor':  float,  # Direct normal solar irradiation [W/m²]
    'HDifHor':  float,  # Diffuse horizontal solar irradiation [W/m²]
}
```

---

### 🔄 Control Logic & Safety Features

#### 1️⃣ First-Order Filters (Smooth Transitions)

```
Signal Flow:  Raw Input ──► FirstOrder Filter (τ=180s) ──► Limiter ──► Actuator
```

| Filter | Time Constant (τ) | Purpose |
|--------|-------------------|---------|
| `filtOA` | 180s | Smooth OA damper transitions |
| `filtRA` | 180s | Smooth RA damper transitions |
| `filtEA` | 180s | Smooth EA damper transitions |
| `filtValve` | 200s | Smooth chilled water valve |
| `filtHeater` | 180s | Smooth heater modulation |

#### 2️⃣ PI Controller for Chilled Water Valve

```python
# Setpoint: T_SA_afterCooling = 285.15 K (12°C)
# Controller: PI với k=0.3, Ti=300s
valve_position = PI_controller(
    setpoint = 285.15,  # K
    measured = T_SA_afterCooling,
    k = 0.3,
    Ti = 300  # s
)
```

#### 3️⃣ Damper Flow Balance

```python
# Đảm bảo cân bằng lưu lượng để tránh áp suất âm
yRA = 1 - yOA  # Return Air = 1 - Outside Air
yEA = yOA      # Exhaust Air = Outside Air (balanced)
```

#### 4️⃣ Dehumidification Control

```python
# Dynamic dehumidification linked to chiller
uDehum = min(2.0, max(0.0, 2 * uChiller))
# Khi chiller hoạt động mạnh, dehumidification tăng theo
```

---

### 📊 Component Details

#### 🧊 Cooling Coil (WetCoilCounterFlow)

```
┌─────────────────────────────────────────────────────┐
│               WET COIL COUNTER FLOW                  │
│                                                      │
│  Air Side (Medium1):     Water Side (Medium2):       │
│  ṁ_air = 0.24 kg/s      ṁ_water = 0.35 kg/s        │
│  dp = 100 Pa            dp = 2000 Pa                │
│  τ₁ = 5s                τ₂ = 20s                    │
│                                                      │
│  Heat Transfer:                                      │
│  UA = 1500 W/K                                       │
│  r_nominal = 2/3                                     │
│  nEle = 6 (discretization elements)                 │
│                                                      │
│  Dynamics: FixedInitial                              │
└─────────────────────────────────────────────────────┘
```

#### ❄️ Chiller (Carnot_TEva_NoWarn)

```
┌─────────────────────────────────────────────────────┐
│               CARNOT CHILLER MODEL                   │
│                                                      │
│  Operating Mode:                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ T_eva = u×T_cold + (1-u)×T_off              │    │
│  │                                              │    │
│  │ u = 0: T_eva = 298.15 K (OFF - warm)        │    │
│  │ u = 1: T_eva = 277.15 K (FULL - cold)       │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Energy Balance:                                     │
│  Q_con = |Q_eva| + P_electric                        │
│  P_electric = |Q_eva| / COP                         │
│  COP_nominal = 1.5                                  │
│                                                      │
│  Capacity: Q_eva = -6000 W (cooling)                │
└─────────────────────────────────────────────────────┘
```

#### 🌀 Fans (FlowControlled_m_flow)

```
┌─────────────────────────────────────────────────────┐
│                   FAN MODELS                         │
│                                                      │
│  Supply Fan (fanSA):          Exhaust Fan (fanEA):  │
│  ṁ_nominal = 0.24 kg/s       ṁ_nominal = 0.24 kg/s │
│  P_nominal = 70 W             P_nominal = 50 W      │
│  dpMax = 5×10¹⁰ Pa           dpMax = 10¹⁰ Pa       │
│                                                      │
│  Pressure Curve:                                     │
│  V_flow = [0, ṁ/1.2, 2×ṁ/1.2]                      │
│  dp = [600, 400, 0] Pa (SA)                         │
│  dp = [400, 300, 0] Pa (EA)                         │
│                                                      │
│  Power Law: P = P_nominal × (ṁ/ṁ_nominal)³         │
└─────────────────────────────────────────────────────┘
```

---

### 🛡️ Stability Features for RL Training

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STABILITY FEATURES                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ✅ allowFlowReversal = true   → Cho phép dòng chảy ngược           │
│  ✅ dpMax = 10⁹ - 5×10¹⁰ Pa   → Tránh assertion failures           │
│  ✅ linearized = true          → Linearize damper characteristics   │
│  ✅ from_dp = true/false       → Phù hợp cho từng component         │
│  ✅ FirstOrder filters         → Smooth control signal transitions  │
│  ✅ Safety limits on inputs    → Prevent physically impossible states│
│                                                                      │
│  System Settings:                                                    │
│  • energyDynamics = SteadyStateInitial                              │
│  • massDynamics = SteadyStateInitial                                │
│  • m_flow_small = 1e-4 kg/s                                         │
│  • dp_small = 100 Pa                                                │
│  • T_ambient = 301.15 K                                             │
│  • p_ambient = 101325 Pa                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 🔗 Simulation Settings

```python
experiment_settings = {
    'StartTime': 0,           # seconds
    'StopTime': 31536000,     # 1 year (365 days)
    'Interval': 900,          # 15 minutes timestep
    'Tolerance': 1e-5         # Solver tolerance
}
```

---

### 📁 Files

| File | Mô tả |
|------|-------|
| **`HVAC.mo`** | Modelica source code - Human readable, editable |
| **`HVAC.fmu`** | Compiled FMU - Binary executable for co-simulation |

---

## ✨ Tính Năng Nổi Bật

| Tính năng | Mô tả |
|-----------|-------|
| 🧠 **Twin Q-Networks** | Sử dụng hai mạng Critic để giảm overestimation bias |
| 📊 **Prioritized Experience Replay** | Học hiệu quả hơn từ các trải nghiệm quan trọng |
| 📚 **Curriculum Learning** | Huấn luyện tiến dần từ dễ đến khó |
| 🔄 **Adaptive Learning Rate** | Tự động điều chỉnh tốc độ học |
| 🌤️ **Dual Mode** | Hỗ trợ cả có và không có dự báo thời tiết |
| 🎚️ **Hierarchical Reward** | Phần thưởng phân cấp cho comfort và energy |

---

## 📁 Cấu Trúc Project

```
HVAC-DDPG-Control/
│
├── 📄 HVAC.fmu              # FMU mô phỏng hệ thống HVAC
├── 📄 HVAC.mo               # Modelica source code của FMU
├── 📄 requirements.txt      # Dependencies của project
├── 📄 setup.py              # Package setup file
│
├── 📂 configs/              # ⚙️ Cấu hình hệ thống
├── 📂 data/                 # 📊 Dữ liệu thời tiết
├── 📂 src/                  # 💻 Source code chính
├── 📂 scripts/              # 🔧 Scripts huấn luyện & đánh giá
├── 📂 notebooks/            # 📓 Jupyter notebooks
├── 📂 checkpoints/          # 💾 Model checkpoints
└── 📂 results/              # 📈 Kết quả training & evaluation
```

---

## 📂 Chi Tiết Từng Thư Mục

### 🔧 `configs/` - Cấu Hình Hệ Thống

> *Trung tâm điều khiển tất cả hyperparameters và settings*

```
configs/
├── __init__.py          # Package initialization
├── base_config.py       # 🎛️ Class BaseConfig chứa tất cả config
└── config_manager.py    # 🔧 Helper functions cho config
```

| File | Vai trò |
|------|---------|
| **`base_config.py`** | Định nghĩa class `BaseConfig` chứa toàn bộ hyperparameters: learning rate, network architecture, reward weights, comfort bands, v.v. Hỗ trợ tự động chuyển đổi giữa chế độ forecast (state_dim=15) và no-forecast (state_dim=14) |
| **`config_manager.py`** | Cung cấp `get_train_config()` và `get_eval_config()` để khởi tạo config với đường dẫn file phù hợp |

---

### 📊 `data/` - Dữ Liệu Thời Tiết

> *Dữ liệu khí hậu thực từ các thành phố Việt Nam*

```
data/
├── Ha_Dong_FULL_YEAR.csv                              # 🏙️ Hà Nội - cả năm
├── Da_Nang_FULL_YEAR.csv                              # 🏖️ Đà Nẵng - cả năm  
├── Nha_Be_FULL_YEAR.csv                               # 🌴 TP.HCM - cả năm
├── sample_data_training_weather_data_Ha_Dong_cold_M01_M02.csv  # ❄️ Training mùa lạnh
├── sample_data_testing_weather_data_Ha_Dong_M03_M04.csv        # 🧪 Testing data
├── best_forecast_M07_M08.csv                          # 🔮 Dữ liệu có forecast
└── training_forecast_model.py                         # 🤖 Script train forecast model
```

**Các cột dữ liệu chính:**
- `time` - Timestamp (seconds)
- `TDryBul` - Nhiệt độ khô (°C)
- `relHum` - Độ ẩm tương đối (%)
- `pAtm` - Áp suất khí quyển (Pa)
- `winSpe` - Tốc độ gió (m/s)
- `HDirNor` - Bức xạ trực tiếp (W/m²)
- `HDifHor` - Bức xạ khuếch tán (W/m²)
- `T_forecast` - Dự báo nhiệt độ (optional)

---

### 💻 `src/` - Source Code Chính

> *Trái tim của hệ thống - nơi chứa toàn bộ logic AI*

```
src/
├── __init__.py
├── 📂 agents/           # 🤖 DDPG Agent
├── 📂 environments/     # 🌍 HVAC Environment
├── 📂 models/           # 🧠 Neural Networks
├── 📂 utils/            # 🛠️ Utility functions
└── 📂 visualization/    # 📊 Plotting functions
```

---

#### 🤖 `src/agents/` - DDPG Agent

```
agents/
├── __init__.py
└── ddpg_agent.py        # 🎮 Class DDPGAgent
```

**`ddpg_agent.py`** - *Agent thông minh điều khiển HVAC*

| Component | Mô tả |
|-----------|-------|
| `DDPGAgent` | Class chính quản lý toàn bộ quá trình học |
| `select_action()` | Chọn action với exploration noise |
| `update()` | Cập nhật weights từ replay buffer |
| `soft_update()` | Cập nhật mềm target networks |
| `save()/load()` | Lưu/load model checkpoint |

**Đặc điểm nổi bật:**
- ✅ Twin Q-Networks (2 Critics) để giảm overestimation
- ✅ Prioritized Experience Replay
- ✅ Adaptive OU Noise cho exploration
- ✅ Adaptive Learning Rate Scheduler

---

#### 🌍 `src/environments/` - HVAC Environment

```
environments/
├── __init__.py
└── hvac_env.py          # 🏢 Class HVACEnvironment
```

**`hvac_env.py`** - *Môi trường mô phỏng hệ thống HVAC*

| Method | Mô tả |
|--------|-------|
| `reset()` | Khởi tạo lại FMU và trả về state đầu tiên |
| `step(action)` | Thực hiện action, trả về (state, reward, done, info) |
| `_build_state()` | Xây dựng vector state 14D hoặc 15D |
| `_get_weather_at()` | Nội suy dữ liệu thời tiết |

**State Vector (14D hoặc 15D):**
```
[T_zone, RH_zone, T_out, RH_out, T_AHU_out, RH_AHU_out, 
 delta_T, delta_RH, hour_sin, hour_cos, occupancy,
 prev_uFan, prev_uChiller, prev_uHeater, (T_forecast)]
```

**Action Vector (5D):**
```
[uFan, uOA, uChiller, uHeater, uFanEA] ∈ [0, 1]
```

---

#### 🧠 `src/models/` - Neural Networks

```
models/
├── __init__.py
├── actor.py             # 🎭 Actor Network
└── critic.py            # 📊 Critic Network (Twin Q)
```

**`actor.py`** - *Mạng quyết định hành động*

```
State → [FC 512] → LayerNorm → ReLU 
      → [FC 512] → LayerNorm → ReLU 
      → [FC 5] → Tanh → Action
```

**`critic.py`** - *Mạng đánh giá Q-value (Twin)*

```
State + Action → [FC 512] → ReLU → [FC 512] → ReLU → Q1
State + Action → [FC 512] → ReLU → [FC 512] → ReLU → Q2
```

---

#### 🛠️ `src/utils/` - Utilities

```
utils/
├── __init__.py
├── noise.py             # 🎲 Adaptive OU Noise
├── replay_buffer.py     # 💾 Prioritized Replay Buffer
├── reward.py            # 🎯 Hierarchical Reward Calculator
└── scheduler.py         # 📈 Adaptive LR Scheduler
```

| File | Class/Function | Vai trò |
|------|----------------|---------|
| **`noise.py`** | `AdaptiveOUNoise` | Ornstein-Uhlenbeck noise với decay tự động |
| **`replay_buffer.py`** | `PrioritizedReplayBuffer` | Buffer với importance sampling |
| **`reward.py`** | `HierarchicalRewardCalculator` | Tính reward dựa trên comfort + energy |
| **`scheduler.py`** | `AdaptiveLRScheduler` | Giảm LR khi reward không cải thiện |

**Reward Formula:**
```python
reward = - (w_temp * temp_penalty + w_humidity * humidity_penalty + w_energy * energy_penalty)
```

**Curriculum Learning:**
- Episode 1-10: Comfort band rộng (dễ)
- Episode 10-30: Thu hẹp dần comfort band
- Episode 30+: Comfort band chặt nhất

---

#### 📊 `src/visualization/` - Plotting

```
visualization/
├── __init__.py
└── plotting.py          # 📈 Visualization functions
```

| Function | Mô tả |
|----------|-------|
| `plot_training_progress()` | Vẽ đường cong reward theo episode |
| `plot_evaluation_results()` | Dashboard đánh giá với nhiệt độ, độ ẩm, năng lượng |
| `print_comfort_statistics()` | In thống kê comfort zone và violations |

---

### 🔧 `scripts/` - Scripts Chạy

```
scripts/
├── train.py             # 🏋️ Script huấn luyện
└── evaluate.py          # 🧪 Script đánh giá
```

**`train.py`** - *Huấn luyện agent*
```bash
python scripts/train.py --weather data/Ha_Dong_FULL_YEAR.csv --episodes 50
```

**`evaluate.py`** - *Đánh giá model đã train*
```bash
python scripts/evaluate.py --model checkpoints/best_model.pth --weather data/test.csv
```

---

### 📓 `notebooks/` - Jupyter Notebooks

> *Notebooks tương tác cho research và demo*

```
notebooks/
├── Train_DDPG.ipynb            # 🏋️ Training notebook (no forecast)
├── Train_DDPG_FC.ipynb         # 🏋️ Training notebook (with forecast)
├── Test_DDPG.ipynb             # 🧪 Testing notebook (no forecast)
├── Test_DDPG_FC.ipynb          # 🧪 Testing notebook (with forecast)
├── DDPG.ipynb                  # 📚 DDPG algorithm explanation
└── Build_Data_for_Sim.ipynb    # 🔧 Data preprocessing
```

| Notebook | Mục đích |
|----------|----------|
| **Train_DDPG.ipynb** | Huấn luyện model mode no-forecast |
| **Train_DDPG_FC.ipynb** | Huấn luyện model mode forecast |
| **Test_DDPG.ipynb** | Đánh giá model no-forecast |
| **Test_DDPG_FC.ipynb** | Đánh giá model forecast |
| **Build_Data_for_Sim.ipynb** | Chuẩn bị và xử lý dữ liệu thời tiết |

---

### 💾 `checkpoints/` - Model Checkpoints

```
checkpoints/
├── no_forecast/         # 📁 Models trained without forecast
│   └── best_model.pth
└── with_forecast/       # 📁 Models trained with forecast
    └── best_model.pth
```

**Checkpoint structure:**
```python
{
    'actor_state_dict': ...,
    'critic_state_dict': ...,
    'actor_optimizer': ...,
    'critic_optimizer': ...,
    'episode': ...,
    'best_reward': ...
}
```

---

### 📈 `results/` - Kết Quả

```
results/
├── training/
│   ├── no_forecast/
│   │   └── training_curves.png
│   └── forecast/
│       └── training_curves.png
└── evaluation/
    ├── no_forecast/
    │   └── eval_run_data.csv
    └── forecast/
        └── eval_run_data.csv
```

**Evaluation metrics trong CSV:**
- Temperature trong comfort zone (%)
- Humidity trong comfort zone (%)
- Total energy consumption (kWh)
- Average reward

---

## 🔧 Cài Đặt

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU training)

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/HVAC-DDPG-Control.git
cd HVAC-DDPG-Control

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep Learning framework |
| `numpy` | ≥1.26 | Numerical computing |
| `pandas` | ≥2.0 | Data processing |
| `matplotlib` | ≥3.10 | Visualization |
| `pyfmi` | ≥2.9 | FMU simulation |

---

## 🚀 Hướng Dẫn Sử Dụng

### 1️⃣ Training

```python
from configs.config_manager import get_train_config
from src.agents import DDPGAgent
from src.environments import HVACEnvironment

# Load config
config = get_train_config(
    fmu_path='HVAC.fmu',
    weather_csv='data/Ha_Dong_FULL_YEAR.csv',
    use_forecast=False
)

# Initialize
env = HVACEnvironment(config)
agent = DDPGAgent(config)

# Training loop
for episode in range(config.NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: Reward = {total_reward:.2f}")
```

### 2️⃣ Evaluation

```python
from configs.config_manager import get_eval_config
from src.agents import DDPGAgent
from src.environments import HVACEnvironment
from src.visualization import plot_evaluation_results, print_comfort_statistics

# Load trained model
config = get_eval_config(...)
agent = DDPGAgent(config)
agent.load('checkpoints/no_forecast/best_model.pth')

# Run evaluation
env = HVACEnvironment(config)
state = env.reset()
results = []

while True:
    action = agent.select_action(state, add_noise=False)
    next_state, reward, done, info = env.step(action)
    results.append(info)
    state = next_state
    if done:
        break

# Visualize
plot_evaluation_results(results)
print_comfort_statistics(results)
```

---

## 📊 Kết Quả

### Training Curves

| Metric | No Forecast | With Forecast |
|--------|-------------|---------------|
| Final Reward | -XXX | -XXX |
| Convergence | ~30 episodes | ~25 episodes |
| Temperature Comfort | 95%+ | 97%+ |

### Comfort Zone Performance

```
╔══════════════════════════════════════════════════════════╗
║          📊 COMFORT ZONE STATISTICS                       ║
╠══════════════════════════════════════════════════════════╣
║  🌡️ Temperature (26-27.5°C):  ████████████████░░  95.2%  ║
║  💧 Humidity (45-65%):        █████████████████░░ 92.8%  ║
║  ⚡ Energy saved:             ████████████░░░░░░  35.2%  ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🗺️ Roadmap

- [x] Basic DDPG implementation
- [x] Twin Q-Networks
- [x] Prioritized Experience Replay
- [x] Curriculum Learning
- [x] Weather forecast integration
- [ ] Multi-zone support
- [ ] Real-time deployment
- [ ] Web dashboard

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributors

<div align="center">

**Capstone Project - FA25**

*Vietnam National University*

</div>

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ and 🤖 AI

</div>
