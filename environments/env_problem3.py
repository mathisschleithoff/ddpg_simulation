import numpy as np 
import random
import math

from gym import Env
from gym.spaces import Discrete, Box

class OptProblem3(Env):
    def __init__(self):
        self.action_space = Box(low = -1., high = 1., shape = (2,), dtype = np.float32)
        self.observation_space = Box(low = -math.inf, high = math.inf, shape = (2,), dtype = np.float32) 
        self.n_steps = 250
        self.state = np.array([10.0, 1.0]).astype(np.float32)

        self.alpha = 0.36
        self.delta = 0.025
        self.gamma = 0.95
        self.rho = 2

        self.cap_lower_bound = 5.0
        self.cap_upper_bound = 15.0
        self.A_lower_bound = 0.5
        self.A_upper_bound = 1.5

    def step(self, action):
        # unscale hours worked into state-dependent action space
        low, high = 0.0001, 0.9999
        labor = np.array([low + (0.5 * (action[1] + 1.0) * (high - low))])
        # unscale investment into state-dependent action space (labor fixed to 1)
        bound = self.state[1] * self.state[0] ** self.alpha * np.array([1.0]) ** (1 - self.alpha)
        low, high = 0.0001*bound, 0.9999*bound
        inv = low + (0.5 * (action[0] + 1.0) * (high - low))
        # clip investment given the actual choice of hours worked (if necessary)
        bound = self.state[1] * self.state[0] ** self.alpha * labor ** (1 - self.alpha)
        inv = np.clip(inv, 0.0001, 0.9999*bound)

        action = np.concatenate((inv, labor)).astype(np.float32)

        reward = math.log(self.state[1] * self.state[0] ** self.alpha * action[1] ** (1 - self.alpha) - action[0]) + self.rho * math.log(1 - action[1])
        reward = float(reward)

        theta = np.random.lognormal(mean=-3.0058, sigma=0.1417, size = (1,))
        A = self.gamma * self.state[1] + theta

        cap = np.array([(1 - self.delta) * self.state[0] + action[0]])

        self.state = np.concatenate((cap, A)).astype(np.float32)
        self.n_steps -= 1        

        if self.n_steps == 0:
            done = True
        else: 
            done = False

        info = {}

        # normalize state before passing to buffer
        low, high = self.cap_lower_bound, self.cap_upper_bound
        norm_cap = 2.0 * ((cap - low) / (high - low)) - 1.0  
        low, high = self.A_lower_bound, self.A_upper_bound
        norm_A = 2.0 * ((A - low) / (high - low)) - 1.0 
        norm_state = np.concatenate((norm_cap, norm_A)).astype(np.float32)
        return norm_state, reward, done, info

    def reset(self, set_seed = False):
        # fix seed (only for prediction purposes)
        if set_seed == True:
            np.random.seed(seed = 1)
            print('Prediction: Seed set to 1')
        self.state = np.array([10.0,1.0]).astype(np.float32)
        self.n_steps = 250

        low, high = self.cap_lower_bound, self.cap_upper_bound
        norm_cap = np.array([2.0 * ((self.state[0] - low) / (high - low)) - 1.0])
        low, high = self.A_lower_bound, self.A_upper_bound
        norm_A = np.array([2.0 * ((self.state[1] - low) / (high - low)) - 1.0]) 
        norm_state = np.concatenate((norm_cap, norm_A)).astype(np.float32)

        return norm_state


