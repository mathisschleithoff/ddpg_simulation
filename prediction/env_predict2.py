import numpy as np
import math

from stable_baselines3 import DDPG
from environments.env_test21 import TestProblem21
from environments.env_test22 import TestProblem22

env1 = TestProblem21()
env2 = TestProblem22()

# define model
model = DDPG.load('./run1_50000_steps')

rl_actions = []
opt_actions = []

A = 1
beta = 0.95
alpha = 0.8
state = env1.reset(set_seed = True)
done = False
cap_upper_bound = 10.0
theta_upper_bound = 1.2

while not done:
    prediction = model.predict(np.reshape(np.array([state]), (1,2)))
    action = prediction[0]
    action = action[0]
    low = 0
    unscaled_cap = low + (0.5 * (state[0] + 1.0) * (cap_upper_bound - low))
    unscaled_theta = low + (0.5 * (state[1] + 1.0) * (theta_upper_bound - low))
    bound = A * unscaled_theta * unscaled_cap**alpha
    low, high = 0.0001*bound, 0.9999*bound
    unscaled_action = low + (0.5 * (action + 1.0) * (high - low))
    rl_actions.append(unscaled_action)
    state, reward, done, info = env1.step(action)

state = env2.reset(set_seed = True)
done = False

while not done:
    action = np.array([beta * alpha * A * state[1] * state[0] ** alpha])
    opt_actions.append(action)
    state, reward, done, info = env2.step(action)

np.savez('./', rl_actions = rl_actions, opt_actions = opt_actions)

print('Deviation of predicted paths:', abs(rl_actions[49] - opt_actions[49]))


