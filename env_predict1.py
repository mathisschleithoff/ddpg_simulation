import numpy as np
import math

from stable_baselines3 import DDPG
from environments.env_problem1 import OptProblem1

env = OptProblem1()

# define model 
model = DDPG.load('./run1_50000_steps')

rl_actions = []

A = 1
alpha = 0.8
state = env.reset()
done = False

while not done:
    prediction = model.predict(np.reshape(np.array([state]), (1,1)))
    action = prediction[0]
    unscaled_state = (0.5 * (state + 1.0) * 10)
    low, high = 0.0001*A*unscaled_state**alpha, 0.9999*A*unscaled_state**alpha
    unscaled_action = low + (0.5 * (action + 1.0) * (high - low))
    rl_actions.append(unscaled_action)
    np.savez('predictions/predictions_1', rl_actions = rl_actions)
    state, reward, done, info = env.step(action)

# deviation between predicted paths after 30 periods
# optimal investment after 30 periods: 0.255
print('Deviation of predicted paths (30 periods):', abs(rl_actions[29] - 0.255))
