# DDPG Simulation
Solving Real Business Cycle Models with Deep Reinforcement Learning

## Description
Application of Deep Reinforcement Learning algorithm Deep Deterministic Policy Gradient (DDPG) on dynamic optimization problems resulting from Real Business Cycle models. DDPG algorithm is reimplemented version of the original provided by Stable Baselines 3 which is based on "Continuous control with deep reinforcement learning" (Lillicrap et al., 2015).

The simulations consist of a training loop (see `training_loop.py` in which the algorithm is applied to the respective problems in several independent training runs. The accuracy of the approximated policy is evaluated by comparing predictions for selected states with those of a benchmark (see `evaluation_metrics.py`). 

## Requirements
- pytorch
- gym 
- numpy

## Usage

# Deterministic Growth Model
`training_loop.py` --train-env-name OptProblem1() --test-env-name OptProblem1() --problem-no 1 --gamma 0.95

# Stochastic Growth Model
`training_loop.py` --train-env-name OptProblem2() --test-env-name OptProblem2() --problem-no 2 --gamma 0.95

# Stochastic Growth Model with Divisible Labor
`training_loop.py` --train-env-name OptProblem3() --test-env-name OptProblem3() --problem-no 3 --gamma 0.99
