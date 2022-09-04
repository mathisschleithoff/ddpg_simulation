import numpy as np 
import random
import math

from gym import Env
from gym.spaces import Discrete, Box

#from stable_baselines3.common.env_checker import check_env
 
class TestProblem22(Env):
    def __init__(self):
        self.action_space = Box(low = -1., high = 1., shape = (1,), dtype = np.float32)
        self.observation_space = Box(low = -1., high = 1., shape = (2,), dtype = np.float32) 
        self.n_steps = 50
        # change starting shock to random
        self.state = np.array([10.0, 1.0]).astype(np.float32)
        self.alpha = 0.8
        self.A = 1
        self.mu = -0.0049752
        self.sigma = 0.099751
        
    def step(self, action):

        reward = math.log(self.A * self.state[1] * self.state[0]**self.alpha - action)
        # float necessary?
        reward = float(reward)

        theta = np.random.lognormal(mean=self.mu, sigma=self.sigma, size = (1,))

        self.state = np.concatenate((action, theta)).astype(np.float32)
        self.n_steps -= 1        

        if self.n_steps == 0:
            done = True
        else: 
            done = False

        info = {}

        return self.state, reward, done, info

    def reset(self, set_seed = False):
        # fix seed (only for prediction purposes)
        if set_seed == True:
            np.random.seed(seed = 1)
            print('Prediction: Seed set to 1')
        self.state = np.array([10.0,1.0]).astype(np.float32)
        self.n_steps = 50
        return self.state
