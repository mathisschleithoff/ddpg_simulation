import numpy as np 
import random
import math

from gym import Env
from gym.spaces import Discrete, Box
 
class OptProblem2(Env):
    def __init__(self):
        self.action_space = Box(low = -1., high = 1., shape = (1,), dtype = np.float32)
        self.observation_space = Box(low = -math.inf, high = math.inf, shape = (2,), dtype = np.float32) 
        self.n_steps = 100
        self.state = np.array([10.0, 1.0]).astype(np.float32)
        self.alpha = 0.8
        self.A = 1
        self.mu = -0.0049752
        self.sigma = 0.099751
        self.cap_upper_bound = 10.0
        self.theta_upper_bound = 1.2

    def step(self, action):

        # unscale investment into state-dependent action space
        bound = self.A * self.state[1] * self.state[0]**self.alpha
        low, high = 0.0001*bound, 0.9999*bound
        action = low + (0.5 * (action + 1.0) * (high - low))

        reward = math.log(self.A * self.state[1] * self.state[0]**self.alpha - action)
        reward = float(reward)

        theta = np.random.lognormal(mean=self.mu, sigma=self.sigma, size = (1,))

        self.state = np.concatenate((action, theta)).astype(np.float32)
        self.n_steps -= 1        

        if self.n_steps == 0:
            done = True
        else: 
            done = False

        info = {}

        # normalize state before passing to buffer
        norm_cap = np.array([2.0 * (self.state[0] / self.cap_upper_bound) - 1.0])
        norm_theta = np.array([2.0 * (self.state[1] / self.theta_upper_bound) - 1.0])
        return np.concatenate((norm_cap, norm_theta)).astype(np.float32), reward, done, info

    def reset(self):
        self.state = np.array([10.0,1.0]).astype(np.float32)
        self.n_steps = 100
        # normalize state before passing to buffer
        norm_cap = np.array([2.0 * (self.state[0] / self.cap_upper_bound) - 1.0])
        norm_theta = np.array([2.0 * (self.state[1] / self.theta_upper_bound) - 1.0])
        return np.concatenate((norm_cap, norm_theta)).astype(np.float32)
