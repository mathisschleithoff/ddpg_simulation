import numpy as np 
import random
import math

from gym import Env
from gym.spaces import Discrete, Box

# environment for application of LQ decision-rules

class TestProblem3(Env):
    def __init__(self):
        self.action_space = Box(low = -1., high = 1., shape = (2,), dtype = np.float32)
        self.observation_space = Box(low = -math.inf, high = math.inf, shape = (2,), dtype = np.float32) 
        self.n_steps = 250
        self.state = np.array([10.0, 1.0]).astype(np.float32)

        self.alpha = 0.36
        self.delta = 0.025
        self.gamma = 0.95
        self.rho = 2

    def step(self, action):

        reward = math.log(self.state[1] * self.state[0] ** self.alpha * action[1] ** (1 - self.alpha) - action[0]) + self.rho * math.log(1 - action[1])
        # float necessary?
        reward = float(reward)

        theta = np.random.lognormal(mean=-3.0058, sigma=0.1417, size = (1,))
        A = self.gamma * self.state[1] + theta

        capital = np.array([(1 - self.delta) * self.state[0] + action[0]])

        self.state = np.concatenate((capital, A)).astype(np.float32)
        self.n_steps -= 1        

        if self.n_steps == 0:
            done = True
        else: 
            done = False

        info = {}

        return self.state, reward, done, info

    def reset(self, set_seed = False):
        # fix seed (only necessary for prediction purposes)
        if set_seed == True:
            np.random.seed(seed = 1)
            print('Prediction: Seed set to 1')
        self.state = np.array([10.0, 1.0]).astype(np.float32)
        self.n_steps = 250

        return self.state
