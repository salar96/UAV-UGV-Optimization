# UAV-UGV-Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **Integrated Path Planning and Facility Location Optimization for UAV-UGV Networks**

This repository implements a novel **Facility Location and Path Optimization (FLPO)** algorithm using deterministic annealing and free energy minimization to solve the integrated path planning and facility location problem for UAV-UGV (Unmanned Aerial Vehicle - Unmanned Ground Vehicle) networks.

## 🚁 Overview

The FLPO algorithm addresses the complex challenge of simultaneously optimizing:
- **UAV route planning** through charging stations to destinations
- **UGV charging station placement** in the operational environment
- **Obstacle avoidance** using geometric path planning
- **Charge constraints** based on UAV battery levels and full charge range (FCR)

### Key Features

- ✅ **Probabilistic routing** using temperature-controlled associations
- ✅ **Obstacle avoidance** via Dijkstra's algorithm on polygon vertices
- ✅ **Deterministic annealing** optimization with deterministic convergence
- ✅ **Multiple algorithm benchmarking** (GA, SA, CEM, PSO, Gurobi)
- ✅ **Interactive visualization** with static plots and animated routes
- ✅ **Normalized coordinate system** for numerical stability

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Benchmarking](#benchmarking)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/salar96/UAV-UGV-Optimization.git
cd UAV-UGV-Optimization

# Install dependencies
pip install -r requirements.txt

# Optional: Install Gurobi for commercial optimization comparison
# pip install gurobipy
```

### Dependencies
Core dependencies are automatically installed via `requirements.txt`:
- `numpy` - Numerical computations
- `scipy` - Optimization algorithms  
- `matplotlib` - Visualization and animation
- `shapely` - Geometric operations for obstacles
- `pyomo` - Mathematical optimization modeling (for Gurobi benchmarks)

## 🚀 Quick Start

### Basic Optimization Example

```python
from UAV_Net import UAV_Net
from annealing import anneal
from viz import plot_drone_routes
import numpy as np

# Define drone missions: ((start_x, start_y), (dest_x, dest_y), charge_level)
drones = [
    ((10.0, 5.0), (45.0, 50.0), 0.7),   # High charge, long distance
    ((20.0, 15.0), (35.0, 35.0), 0.6),  # Medium charge, moderate distance
    ((5.0, 30.0), (25.0, 5.0), 0.4),    # Low charge, short distance
]

# Initialize charging station positions
N_stations = 3
init_ugv = np.array([[0.3, 0.7], [0.6, 0.4], [0.8, 0.8]])

# Set optimization parameters
fcr = 25.0  # Full Charge Range
ugv_factor = 0.0  # UGV movement cost factor

# Create optimization network
uav_net = UAV_Net(drones, N_stations, init_ugv, blocks=None, 
                  ugv_factor=ugv_factor, fcr=fcr, distance="euclidean")

# Run deterministic annealing optimization
Y_s, Betas = anneal(
    obj=uav_net.objective,
    init_stations=uav_net.stations,
    bounds=uav_net.bounds,
    beta_init=1e-4,
    beta_f=1e4,
    alpha=2.0,
    purturb=1e-6,
    method='powell',
    verbos=True
)

# Get optimized routes
from FLPO import calc_associations, calc_routs
P_ss = calc_associations(uav_net.D_ss, Betas[-1])
routes = calc_routs(P_ss)

# Visualize results
plot_drone_routes(drones, Y_s[-1], blocks=None, routes=routes, 
                  fcr=fcr, ugv_factor=ugv_factor)
```



## 📊 Algorithm Details

### FLPO (Facility Location and Path Optimization)

The FLPO algorithm combines:

1. **Free Energy Minimization**: Uses probabilistic associations between drones and charging stations
2. **Deterministic Annealing**: Temperature-controlled optimization from β_init=1e-4 to β_f=1e4
3. **Smooth Penalty Functions**: Handles charge constraints without hard boundaries
4. **Coordinate Normalization**: Maps all coordinates to [0,1] for numerical stability

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_init` | 1e-4 | Initial temperature parameter |
| `beta_f` | 1e4 | Final temperature parameter |
| `alpha` | 2.0 | Temperature growth rate |
| `fcr` | 25.0 | Full Charge Range for UAVs |
| `purturb` | 1e-6 | Random perturbation in optimization |

### Convergence Criteria
- Relative cost change < 1e-4
- Maximum iterations reached
- Temperature ceiling achieved

## 🏆 Benchmarking

Compare FLPO against state-of-the-art optimization algorithms:

```python
# Run benchmark comparison (see benchmark.ipynb)
algorithms = ['FLPO', 'GA', 'SA', 'CEM', 'PSO', 'Gurobi']
# Results automatically saved to Benchmark/ directory
```

### Supported Benchmark Algorithms
- **Genetic Algorithm (GA)** - `Benchmark/GA.py`
- **Simulated Annealing (SA)** - `Benchmark/SA.py` 
- **Cross-Entropy Method (CEM)** - `Benchmark/CEM.py`
- **Particle Swarm Optimization (PSO)** - `Benchmark/PSO.py`
- **Gurobi MIP Solver** - `Benchmark/GurobiSolver.py` (requires license)

## 🎨 Visualization

### Static Visualization
```python
from viz import plot_drone_routes
plot_drone_routes(drones, stations, blocks, routes, fcr, ugv_factor, save_=True)
```

### Animated Visualization  
```python
from animator import animate_drone_routes
animate_drone_routes(drones, stations, blocks, routes, fcr, ugv_factor,
                     animation_speed=2.0, fps=30, save_path="animation.gif")
```

### Obstacle Creation
```python
from utils import create_block

# Create various obstacle shapes
hexagon = create_block("hexagon", center=(30.0, 30.0), length=3.0)
square = create_block("square", center=(15.0, 25.0), length=3.5, distortion="rotated")
triangle = create_block("triangle", center=(30.0, 10.0), length=2.0, distortion="skewed")
```

## 📁 Project Structure

```
UAV-UGV-Optimization/
├── 📓 main.ipynb              # Primary research notebook
├── 📓 benchmark.ipynb         # Algorithm comparison experiments
├── 🐍 UAV_Net.py             # Core optimization network class
├── 🐍 FLPO.py                # Free energy and association functions
├── 🐍 annealing.py           # Deterministic annealing implementation
├── 🐍 viz.py                 # Static visualization functions
├── 🐍 animator.py            # Animation visualization
├── 🐍 ObstacleAvoidance.py   # Geometric path planning
├── 🐍 normalizers.py         # Coordinate normalization utilities
├── 🐍 utils.py               # General utility functions
├── 📁 Benchmark/             # Comparison algorithms
│   ├── GA.py, SA.py, CEM.py, PSO.py, GurobiSolver.py
│   └── N{drones}_M{stations}_seed{seed}  # Results cache
├── 📋 requirements.txt       # Dependencies
├── 📋 .github/copilot-instructions.md  # AI agent guidelines
└── 📖 README.md              # This file
```

## 🧪 Examples and Notebooks

### Jupyter Notebooks
1. **`main.ipynb`** - Complete optimization workflow with visualization
2. **`benchmark.ipynb`** - Algorithm performance comparison with statistical analysis

### Example Scenarios
- Multi-drone missions with varying charge levels
- Complex obstacle environments with geometric shapes
- Large-scale networks with 10+ drones and 5+ stations

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style conventions
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{11314119,
  author={Basiri, Salar and Tiwari, Dhananjay and Hassani, Ghazal and Papachristos, Christos and Salapaka, Srinivasa},
  journal={IEEE Access}, 
  title={A Maximum Entropy Approach to Joint Obstacle-Aware Routing and Facility Location in UAV Networks Under Energy Constraints}, 
  year={2025},
  volume={13},
  number={},
  pages={216694-216704},
  keywords={Autonomous aerial vehicles;Costs;Routing;Entropy;Batteries;Logistics;Metaheuristics;Land vehicles;Collision avoidance;Wireless communication;Facility location;maximum entropy principle;multi-objective planning;optimization;path planning;routing;uncrewed aerial vehicles (UAV)},
  doi={10.1109/ACCESS.2025.3647827}}

```


## ❓ Support

For questions, issues, or feature requests:
- 📧 Create an [Issue](https://github.com/salar96/UAV-UGV-Optimization/issues)
- 💬 Start a [Discussion](https://github.com/salar96/UAV-UGV-Optimization/discussions)
- 📖 Check the [Wiki](https://github.com/salar96/UAV-UGV-Optimization/wiki)

---

<div align="center">
  <strong>Happy Optimizing! </strong>
</div>
