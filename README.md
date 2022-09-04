# DDPG Simulation
Solving Real Business Cycle Models with Deep Reinforcement Learning

## Description
Application of Deep Reinforcement Learning algorithm Deep Deterministic Policy Gradient (DDPG) to solve dynamic optimization problems resulting from Real Business Cycle models. DDPG algorithm is provided by [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) which is based on ["Continuous control with deep reinforcement learning"](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015). 

The simulations consist of a training loop (see `training_loop.py`) in which the algorithm is applied to the respective problems in several independent training runs. The accuracy of the approximated policy is evaluated by comparing predictions for selected states with those of a benchmark (see `evaluation_metrics.py`). 

## Requirements
- pytorch
- gym 
- numpy

## Usage

### Deterministic Growth Model
Apply DDPG Algorithm + Evaluation:<br>
`training_loop.py` train-env = OptProblem1(), test-env = TestProblem1(), problem-no = 1, gamma = 0.95, learning-rate = linear_schedule(1e-3,1e-5)

View path for economic variables predicted by trained model:<br>
1.`env_predict1.py`, 2.`predict_paths.R`

### Stochastic Growth Model
Apply DDPG Algorithm + Evaluation:<br>
`training_loop.py` train-env = OptProblem2(), test-env = TestProblem21(), problem-no = 2, gamma = 0.95, learning-rate = linear_schedule(1e-3,1e-5)

View path for economic variables predicted by trained model:<br>
1.`env_predict2.py`, 2.`predict_paths.R`

### Stochastic Growth Model with Divisible Labor
Apply DDPG Algorithm + Evaluation:<br>
`training_loop.py` train-env = OptProblem3(), test-env = OptProblem3(), problem-no = 3, gamma = 0.99, learning-rate = 1e-4

View path for economic variables predicted by trained model:<br>
1.`env_predict3.py`, 2.`predict_paths.R`
