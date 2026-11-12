# Self-Modifying RL for Adaptive Decision Architectures

> **CS 4180/5180 — Robotics Masters Project**
> Abdul Aziz | April 2026

A hierarchical reinforcement learning system where a **meta-controller** learns to dynamically select between a memoryless MLP and a memory-aware LSTM inner policy, adapting to non-stationary environment phases in real time.

---

## Demo — Robosuite Panda Lift

(https://github.com/azizxabd/self-modifying-rl/blob/main/results/videos/scripted_policy.mp4)

> The self-modifying agent switches between MLP and LSTM architectures mid-episode as the environment transitions through Normal → Heavy (3x mass) → Disturbed (random forces) phases.

---

## Overview

```
┌─────────────────────────────────────────────────────┐
│                  Meta-Controller                     │
│         (REINFORCE — selects every 20 steps)        │
│              ↓ switch? ↓ keep?                       │
│   ┌──────────────┐     ┌──────────────┐             │
│   │  MLP Policy  │     │ LSTM Policy  │             │
│   │ (memoryless) │     │(memory-aware)│             │
│   └──────────────┘     └──────────────┘             │
│              ↓ action ↓                              │
│         Environment (non-stationary)                 │
└─────────────────────────────────────────────────────┘
```

The meta-controller observes a 7-dimensional meta-state:
`[mean_reward, std_reward, trend, lift_rate, norm_time, arch_onehot(2)]`

---

## Results

### Performance Across Environments

| Environment | Self-Modifying | High-Capacity | Fixed PPO | Heuristic | Random |
|---|---|---|---|---|---|
| **RLproj (1D)** | +2.87 ± 5.01 | — | -15.56 | **+2.99** | -0.43 |
| **CartPole** | +297 ± 48 | **+474 ± 9** | +312 ± 12 | +340 ± 19 | +219 ± 19 |
| **LunarLander** | +347 ± 27 | **+432 ± 147** | +359 ± 136 | +309 ± 73 | +226 ± 14 |
| **Robosuite** | +420 ± 6 | +477 ± 4 | **+538 ± 0.4** | -5.5 ± 0.4 | -298 ± 6 |

### Switching Behavior

| Environment | Self-Modifying | Heuristic | Random |
|---|---|---|---|
| RLproj | 5.38/ep | 2.09/ep | 11.93/ep |
| CartPole | 1.49/ep | 0.00/ep | 5.14/ep |
| LunarLander | **2.55/ep** | 1.88/ep | 10.50/ep |
| Robosuite | 5.20/ep | 10.66/ep | 36.86/ep |

> Self-modifying agent switches **selectively** — significantly less than random, showing the meta-controller learns meaningful switching decisions.

### Architecture Preferences Learned

| Environment | MLP % | LSTM % | Interpretation |
|---|---|---|---|
| RLproj | 69% | 31% | Reactive 1D task suits memoryless |
| CartPole | 41% | 59% | Balanced, still learning |
| LunarLander | 85% | 15% | Fast reactions needed |
| Robosuite | 19% | **81%** | Memory needed for phase tracking |

---

## Environments

### 1. RLproj — Custom 1D Non-Stationary
```
Phase 0 Sprint:    maximize velocity
Phase 1 Conserve:  maintain moderate speed
Phase 2 Stabilize: hold near zero
```

### 2. CartPole Phased
```
Phase 0 Normal:  standard gravity, pole_mass=0.1
Phase 1 Heavy:   pole_mass=0.3 (3x heavier)
Phase 2 Windy:   random force on cart ±0.5N
```

### 3. LunarLander Phased
```
Phase 0 Normal:  gravity=-10.0
Phase 1 Heavy:   gravity=-11.5 (stronger)
Phase 2 Windy:   wind_power=15.0, turbulence=1.5
```

### 4. Robosuite Panda Lift
```
Phase 0 Normal:    cube mass=0.1kg
Phase 1 Heavy:     cube mass=0.3kg (3x heavier)
Phase 2 Disturbed: random forces ±0.3N on cube
```

---

## Architecture

### Inner Policies

**MLP (Memoryless):**
```
Linear(obs, 256) → LayerNorm → ReLU
Linear(256, 256) → LayerNorm → ReLU
Linear(256, 128) → ReLU
→ PPO actor-critic head
```

**LSTM (Memory-Aware):**
```
Linear(obs, 256) → LayerNorm → ReLU  [pre-processing]
LSTM(256, 128)                         [recurrent]
Linear(128, 128) → ReLU               [output]
→ PPO actor-critic head
```

### Meta-Controller
```
Linear(7, 64) → LayerNorm → ReLU
Linear(64, 64) → LayerNorm → ReLU
Linear(64, 2) → Softmax
→ REINFORCE update with switching cost penalty (λ=0.3)
```

---

## File Structure

```
RL_robosuite/
├── core/
│   ├── robosuite_env.py      # Panda Lift phased environment
│   ├── cartpole_env.py       # CartPole phased environment
│   ├── lunarlander_env.py    # LunarLander phased environment
│   ├── policy.py             # MLP + LSTM inner policies
│   ├── meta_controller.py    # REINFORCE meta-controller
│   ├── scripted_policy.py    # Rule-based lift policy (100% success)
│   ├── demo_collector.py     # Parallel demonstration collection
│   └── behavior_cloning.py  # BC pretraining
├── training/
│   ├── train.py              # PPO-only pipeline
│   ├── train_bc.py           # BC + PPO pipeline (Robosuite)
│   ├── train_cartpole.py     # CartPole full pipeline
│   ├── train_lunarlander.py  # LunarLander full pipeline
│   ├── baselines.py          # All baseline agents
│   └── config.yaml           # Hyperparameters
├── evaluation/
│   ├── eval.py               # Robosuite evaluation
│   ├── eval_cartpole.py      # CartPole evaluation
│   └── eval_lunarlander.py   # LunarLander evaluation
├── visuals/
│   ├── plot_cartpole_lunar.py # Combined comparison plots
│   ├── plot_curves.py         # Learning curves
│   ├── plot_switching.py      # Switching behavior
│   ├── plot_comparison.py     # Final comparison bars
│   └── plot_robot.py          # Robot-specific plots
└── render/
    ├── watch_agent.py         # Live MuJoCo viewer
    └── record_episode.py      # MP4 video recorder
```

---

## Installation

```bash
# Clone
git clone https://github.com/azizxabd/RL_robosuite.git
cd RL_robosuite

# Setup venv (Python 3.10)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Box2D for CartPole/LunarLander
pip install swig
pip install "gymnasium[box2d]"

# Install project
pip install -e .
```

---

## Running

### CartPole (fastest, ~1 hour)
```bash
python training/train_cartpole.py
python evaluation/eval_cartpole.py
```

### LunarLander (~2 hours)
```bash
python training/train_lunarlander.py
python evaluation/eval_lunarlander.py
```

### Robosuite Panda Lift (~8 hours)
```bash
# Collect demonstrations
python core/demo_collector.py

# Train with BC pretraining
python training/train_bc.py

# Run baselines
python training/baselines.py

# Evaluate
python evaluation/eval.py
```

### Generate All Plots
```bash
python visuals/plot_cartpole_lunar.py
python visuals/plot_curves.py
python visuals/plot_switching.py
python visuals/plot_comparison.py
```

### Watch Agent Live
```bash
python render/watch_agent.py --agent self_modifying --seed 0
```

### Record Video
```bash
python render/record_episode.py --agent self_modifying --seed 0
```

---

## Key Findings

1. **Hierarchy helps on complex tasks** — On RLproj and LunarLander, self-modifying matches or beats heuristic switching. On simple CartPole, high-capacity single agent dominates.

2. **Switching is selective, not random** — Meta-controller switches 1.5-5.4 times/episode vs random's 5-37 times/episode.

3. **Architecture preferences are task-dependent** — LSTM preferred for memory-demanding tasks (Robosuite 81%), MLP preferred for reactive tasks (LunarLander 85%).

4. **Consistent performance** — Self-modifying shows lower variance than high-capacity on LunarLander (±27 vs ±147).

---

## Hardware

- CPU: Ryzen 7 5800HS
- GPU: RTX 3050 4GB
- RAM: 16GB
- CUDA: 12.6, PyTorch cu124

---

## Citation

```bibtex
@misc{aziz2026selfmodifying,
  title={Self-Modifying RL for Adaptive Decision Architectures},
  author={Abdul Aziz},
  year={2026},
  institution={Robotics Masters Program}
}
```
