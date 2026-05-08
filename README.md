# Deep Reinforcement Learning for Flappy Bird

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Open%20Source-green)](LICENSE)
[![DQN Implementation](https://img.shields.io/badge/algorithm-DQN-orange)](https://arxiv.org/abs/1312.5602)

A comprehensive Deep Reinforcement Learning project implementing a Deep Q-Network (DQN) using PyTorch and Gymnasium. This project trains an autonomous AI agent to play Flappy Bird through self-directed learning.

---

## 🎯 Overview

This project demonstrates practical reinforcement learning by training an intelligent game-playing agent using Deep Q-Learning. The agent learns optimal gameplay strategies by interacting with the Flappy Bird environment, storing experiences in replay memory, and continuously improving decision-making through neural network optimization.

### Key Learning Concepts Implemented
- **Deep Q-Learning (DQN)**: Neural network approximation of Q-values
- **Experience Replay**: Stabilized training through random sampling of past experiences
- **Target Network**: Improved convergence through periodic synchronization
- **Epsilon-Greedy Exploration**: Balanced exploration and exploitation strategies

---

## ✨ Features

- ✅ Deep Q-Network (DQN) architecture
- ✅ Experience Replay Memory buffer
- ✅ Target Network synchronization
- ✅ Epsilon-Greedy exploration strategy
- ✅ Real-time training visualization
- ✅ Reward tracking and logging
- ✅ Model checkpointing and loading
- ✅ YAML-based hyperparameter configuration
- ✅ Automated Flappy Bird gameplay
- ✅ Training graphs and performance metrics

---

## 🛠️ Technology Stack

| Technology | Purpose |
|:---|:---|
| **Python** | Core programming language |
| **PyTorch** | Neural network framework & deep learning |
| **Gymnasium** | Reinforcement learning environment simulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Training visualization & graphing |
| **YAML** | Configuration management |

---

## 📁 Project Structure

```
.
├── agent.py                  # Main training & testing pipeline
├── dqn.py                    # Deep Q-Network architecture
├── experience_replay.py       # Experience replay memory implementation
├── hyperparameters.yml        # Training hyperparameters
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── runs/                      # Training outputs (auto-generated)
    ├── flappybird1.log       # Training logs
    ├── flappybird1.png       # Reward & epsilon decay graphs
    └── flappybird1.pt        # Saved model weights
```

### File Descriptions

| File | Description |
|:---|:---|
| `agent.py` | Main entry point; contains training loop, evaluation, and agent management |
| `dqn.py` | Deep Q-Network implementation with forward pass and loss computation |
| `experience_replay.py` | Replay buffer for storing and sampling experiences (s, a, r, s', done) |
| `hyperparameters.yml` | Configurable training parameters (learning rate, batch size, etc.) |
| `runs/` | Output directory for logs, checkpoints, and performance graphs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda
- 2GB+ disk space for model checkpoints and logs

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/deep-reinforcement-learning-flappy-bird.git
cd deep-reinforcement-learning-flappy-bird
```

#### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv dqnenv
dqnenv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv dqnenv
source dqnenv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch gymnasium matplotlib numpy pyyaml flappy-bird-gymnasium
```

---

## 📖 Usage

### Training the Agent

Start training with:

```bash
python agent.py flappybird1 --train
```

**What happens during training:**
- Agent plays thousands of episodes
- Rewards are logged in real-time
- Performance graphs are generated
- Best model checkpoints are automatically saved

**Customization:**
Modify `hyperparameters.yml` to adjust:
- Learning rate
- Replay memory size
- Epsilon decay rate
- Discount factor (gamma)
- Batch size
- Target network update frequency

### Running a Trained Agent

To watch the trained agent play automatically:

```bash
python agent.py flappybird1
```

Ensure `render=True` is set in `agent.py` to visualize gameplay.

### Testing Specific Models

```bash
python agent.py flappybird1 --model runs/flappybird1.pt
```

---

## 📊 Training Outputs

All outputs are saved to the `runs/` directory automatically.

**Example output structure:**
```
runs/
├── flappybird1.log    # Training logs with episode rewards
├── flappybird1.png    # Graphs: cumulative reward & epsilon decay
└── flappybird1.pt     # Best model weights (PyTorch format)
```

### Output Files

| File | Purpose |
|:---|:---|
| `.log` | Text log with episode-by-episode rewards and training metrics |
| `.png` | Two-panel visualization: (1) cumulative rewards over time, (2) epsilon decay curve |
| `.pt` | PyTorch model checkpoint; can be loaded for testing or resuming training |

### Interpreting Results

- **Increasing reward trend** → Agent is learning and improving
- **Decaying epsilon curve** → Agent shifts from exploration to exploitation
- **Flat plateau** → May indicate overfitting or local optima

---

## ⚙️ Configuration

All hyperparameters are defined in `hyperparameters.yml`:

```yaml
learning_rate: 0.0001
discount_factor: 0.99
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995
batch_size: 32
replay_memory_size: 10000
target_update_frequency: 1000
episodes: 5000
render: false
```

### Parameter Guidelines

| Parameter | Recommended Range | Effect |
|:---|:---|:---|
| `learning_rate` | 0.00001 - 0.001 | Lower = slower learning, more stable |
| `batch_size` | 16 - 64 | Higher = more stable, slower per-step |
| `epsilon_decay` | 0.99 - 0.999 | Controls exploration→exploitation transition |
| `replay_memory_size` | 5000 - 100000 | Larger = more diverse experience samples |

---

## 🧠 Reinforcement Learning Concepts

### Deep Q-Learning (DQN)

The agent learns an action-value function Q(s,a) using a neural network instead of tabular storage. The network predicts expected future rewards for each action in a given state.

**Bellman Equation:**
```
Q(s, a) = r + γ × max Q(s', a')
```

### Experience Replay

Instead of learning from sequential experiences (which are highly correlated), the agent stores transitions in a buffer and samples random batches. This:
- Reduces variance in learning
- Improves data efficiency
- Stabilizes training

### Target Network

A separate "target" network with frozen weights is used to compute target Q-values. This network is periodically synchronized with the main network, reducing harmful oscillations during training.

### Epsilon-Greedy Strategy

- **Exploration (probability ε)**: Take random actions to discover new strategies
- **Exploitation (probability 1-ε)**: Select actions with highest Q-values

Epsilon decays over time, shifting the agent from exploration to exploitation.

---

## 📈 Performance Metrics

The agent's learning progress is measured by:

1. **Cumulative Episode Reward**: Sum of rewards per episode
2. **Average Reward (windowed)**: Rolling average over recent episodes
3. **Exploration Rate (ε)**: Probability of random action; should decrease over time

Typical learning curve:
- Episodes 1-500: High variance, rapid improvement
- Episodes 500-2000: Steady improvement, decreasing variance
- Episodes 2000+: Convergence to near-optimal policy

---

## 🔮 Future Improvements

Potential enhancements to explore:

- **Double DQN**: Reduces overestimation bias in Q-values
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions more frequently
- **CNN-Based Input**: Learn directly from pixel observations
- **TensorBoard Integration**: Real-time training visualization
- **Checkpoint Resume**: Continue training from saved checkpoints
- **Multi-Environment Support**: Train on multiple Flappy Bird variants
- **Distributional RL**: Learn full return distribution, not just expected value

---

## 📋 Requirements

See `requirements.txt` for complete dependency list:

```
torch>=2.0.0
gymnasium>=0.27.0
flappy-bird-gymnasium>=0.3.0
matplotlib>=3.5.0
numpy>=1.20.0
pyyaml>=6.0
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: GPU out of memory

**Solution:** Reduce `batch_size` in `hyperparameters.yml` (try 16 or 8)

### Issue: Agent not improving

**Possible causes:**
- Learning rate too high → Reduce to 0.00001
- Epsilon decay too slow → Try 0.99
- Replay buffer too small → Increase to 10000+

### Issue: Training is very slow

**Solutions:**
- Use GPU: Ensure PyTorch is installed with CUDA support
- Reduce `episodes` for testing
- Increase `batch_size` (if GPU memory allows)

---

## 📚 Resources & References

### Academic Papers
- [Playing Atari with Deep Reinforcement Learning (Original DQN Paper)](https://arxiv.org/abs/1312.5602)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Flappy Bird Gymnasium](https://github.com/Tasafe/flappy-bird-gymnasium)

### Learning Resources
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [DeepMind Reinforcement Learning Course](https://www.youtube.com/watch?v=ISk80iLg3o0)

---

## 📝 License

This project is open-source and available for educational and research purposes. See LICENSE file for details.

---

## 👤 Author

**Vishal Saha**

Contributions and feedback are welcome! Feel free to open issues and pull requests.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation in `README.md`
- Review the troubleshooting section above

---

## 🎓 Educational Value

This project is designed to teach:
- Practical implementation of DQN from scratch
- PyTorch neural network development
- Reinforcement learning principles
- Hyperparameter tuning and optimization
- Training monitoring and visualization

Perfect for students and researchers learning deep RL!

---

**Last Updated:** January 2025
**Project Status:** Active Development
