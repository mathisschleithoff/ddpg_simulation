import numpy as np 
import random
import math

from gym import Env
from gym.spaces import Discrete, Box
 
class OptProblem1(Env):
    def __init__(self):
        self.action_space = Box(low = -1., high = 1., shape = (1,), dtype = np.float32)
        self.observation_space = Box(low = -math.inf, high = math.inf, shape = (1,), dtype = np.float32) 
        self.n_steps = 100
        self.state = np.array([10.0]).astype(np.float32)
        self.alpha = 0.8
        self.A = 1

    def step(self, action):

        # unscale investment into state-dependent action space
        low, high = 0.0001*self.A*self.state**self.alpha, 0.9999*self.A*self.state**self.alpha
        action = low + (0.5 * (action + 1.0) * (high - low))

        reward = math.log(self.A*self.state**self.alpha - action)
        reward = float(reward)

        self.state = np.array(action).astype(np.float32)
        self.n_steps -= 1        

        if self.n_steps == 0:
            done = True
        else: 
            done = False

        info = {}

        # scale state into normalized interval before passing to buffer
        return 2.0 * (self.state / 10.0) - 1.0, reward, done, info

    def reset(self):
        self.state = np.array([10.0]).astype(np.float32)
        self.n_steps = 100
        # scale state before passing to buffer
        return 2.0 * (self.state / 10.0) - 1.0
