# HVAC Control with Deep Reinforcement Learning (HVAC Demo)

This repository presents a demonstration system for intelligent Heating, Ventilation, and Air Conditioning (HVAC) control using Deep Reinforcement Learning (DRL), designed specifically for Vietnamese building climate conditions.  
The project integrates a physics-based digital twin with online control inference and an interactive visualization interface, moving beyond purely offline simulation toward practical deployment.

---

## Project Objectives

- Demonstrate the practical applicability of Deep Reinforcement Learning in HVAC control systems
- Optimize the trade-off between:
  - Thermal comfort
  - Indoor Air Quality (including carbon dioxide concentration and relative humidity)
  - Energy efficiency
- Provide a modular system architecture that connects simulation, control inference, and visualization

---

## Core Technologies

- Deep Reinforcement Learning  
  - Continuous control algorithms (Deep Deterministic Policy Gradient as baseline, extensible to Proximal Policy Optimization or Soft Actor-Critic)
- Physics-Based HVAC Modeling  
  - Modelica Buildings Library  
  - Exported as a Functional Mock-up Unit (FMU)
- Backend and Control Layer  
  - Python version 3.10  
  - FastAPI for RESTful application programming interfaces  
  - Uvicorn as the ASGI server
- Frontend Visualization  
  - Streamlit interactive dashboard
- Deployment and Connectivity  
  - Local or virtual machine execution  
  - Optional remote access using secure tunneling (for example, ngrok)

---

## System Architecture Overview

The system follows a supervisory control architecture consisting of four main components:

1. FMU-Based HVAC Digital Twin  
   - Simulates HVAC equipment dynamics, indoor thermal behavior, and indoor air quality indicators

2. Deep Reinforcement Learning Supervisory Controller  
   - Observes system states from the digital twin  
   - Computes continuous control actions for HVAC actuators

3. FastAPI Control Interface  
   - Provides stateless REST endpoints for simulation stepping, control synchronization, and data exchange

4. Streamlit Visualization Interface  
   - Displays real-time system states, control actions, comfort metrics, and energy-related indicators

---

## Main Features

- Real-time visualization of HVAC system states such as zone temperature, relative humidity, and carbon dioxide concentration
- Continuous control signals including fan speed, outdoor air damper ratio, and heating or cooling commands
- Modular application programming interfaces enabling easy extension or replacement of control strategies
- Extensible reinforcement learning pipeline supporting different algorithms and reward designs
- Supports local demonstration as well as remote access via tunneling services

---

## Repository Structure (Typical)

HVAC_Demo/
├─ backend/
│ ├─ api.py # FastAPI application entry point
│ ├─ app.py # Streamlit user interface
│ ├─ controllers/ # Control logic (reinforcement learning, rule-based, model predictive control)
│ ├─ rl/ # Reinforcement learning agents (actor, critic, replay buffer)
│ ├─ fmu/ # Functional Mock-up Unit models and wrappers
│ └─ utils/ # Data preprocessing and visualization utilities
├─ data/
│ ├─ weather/ # Outdoor weather data
│ └─ logs/ # Simulation and control logs
├─ models/
│ ├─ ddpg_best.pth # Best-performing Deep Deterministic Policy Gradient model
│ └─ checkpoints/ # Periodic training checkpoints
├─ requirements.txt
└─ README.md

## Installation

### Step 1: Create a Python environment (recommended Python version 3.10)
```
python -m venv .venv

```
### Step 2: Install dependencies
```
pip install -r requirements.txt

```

### Start the backend control server
```
python -m uvicorn backend.api:app --reload --port 8000

```
### Start the Streamlit visualization interface
```
streamlit run backend/app.py

```
The Streamlit interface will be available at:

```
http://localhost:8501
```

## Remote Access (Optional)
To expose the Streamlit interface for remote demonstration:

```
ngrok http 8501

```
Use the generated public URL to access the demo remotely.


## Performance Metrics and Evaluation

The system supports monitoring and evaluation of:

- Thermal comfort compliance
- Indoor air quality compliance
- Energy consumption proxies
- Control stability and smoothness
- Comparison between rule-based control and deep reinforcement learning

---

## Research Context

This project supports an academic capstone focused on applying deep reinforcement learning to HVAC supervisory control under Vietnamese climate conditions, combining physics-based digital twins with online decision-making.

---

## Future Work

- Integration of model predictive control with explicit constraints
- Extension to multi-zone building scenarios
- Containerized deployment with service separation
- Long-term monitoring of control policy robustness

---

## License

This project is intended for research and educational purposes.  
Add an appropriate license if the project is released as open source.

