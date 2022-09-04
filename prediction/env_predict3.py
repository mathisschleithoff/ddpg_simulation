import numpy as np
import math

from stable_baselines3 import DDPG
from environments.env_problem3 import OptProblem3
from environments.env_problem3_lq import TestProblem3

env1 = OptProblem3()
env2 = TestProblem3()

# define model
model = DDPG.load('run1_50000_steps')

rl_labor = []
rl_inv = []
lq_labor = []
lq_inv = []

state = env1.reset(set_seed = True)
done = False
n_steps = 0
rl_score = 0
alpha = 0.36

while not done:
    prediction = model.predict(np.reshape(np.array([state]), (1,2)), deterministic = True)
    action = prediction[0]
    action = action[0]
    labor = action[1]
    low, high = 0.0001,0.9999
    labor = np.array([low + (0.5 * (labor + 1.0) * (high - low))])
    low,high = 5.0, 15.0
    cap = low + (0.5 * (state[0] + 1.0) * (high - low))
    low,high = 0.5,1.5
    A = low + (0.5 * (state[1] + 1.0) * (high - low))

    bound = A * cap ** alpha * np.array([1.0]) ** (1 - alpha)
    low, high = 0.0001*bound, 0.9999*bound
    inv = low + (0.5 * (action[0] + 1.0) * (high - low))
    bound = A * cap ** alpha * labor ** (1 - alpha)
    inv = np.clip(inv, 0.0001, 0.9999*bound)

    rl_labor.append(labor)
    rl_inv.append(inv)
    state, reward, done, info = env1.step(action)
    rl_score += reward * 0.99 ** n_steps
    n_steps += 1

j = np.array([[-0.7883363, 1.3277898, -0.022197655],
              [0.1501221, 0.2291501, -0.006859907]])
state = env2.reset(set_seed = True)
done = False
lq_score = 0
n_steps = 0

while not done:
    f = np.array([1, state[1], state[0]])
    action = np.matmul(j,f)
    lq_labor.append(action[1])
    lq_inv.append(action[0])
    state, reward, done, info = env2.step(action)
    lq_score += reward * 0.99 ** n_steps
    n_steps += 1

print(rl_score)
print(lq_score)
np.savez('./', rl_labor = rl_labor, rl_inv = rl_inv,
            lq_labor = lq_labor, lq_inv = lq_inv)
