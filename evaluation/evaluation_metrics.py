import numpy as np
import math

from environments.env_problem3_lq import TestProblem3

class EvalProb1():
    def __init__(self):
        self.alpha = 0.8
        self.beta = 0.95
        self.A = 1
        self.cap_upper_bound = 10.0
        # relevant states
        self.eval_capital = [10.0, 4.8, 2.66, 1.66, 1.14, 0.85, 0.66, 0.47, 0.31, 0.25]
        self.rel_dev = 0
        self.rel_dev10 = 0

    def evaluate(self, model, eval_env):
        # model 1: determine mean absolute fractional difference (MAFD) for relevant states
        self.rel_dev = 0
        self.rel_dev10 = 0
        for cap in self.eval_capital:
            # scale capital
            scaled_state = np.array([2.0 * (cap / self.cap_upper_bound) - 1.0]).astype(np.float32)
            prediction = model.predict(np.reshape(np.array([scaled_state]), (1,1)))
            scaled_action = prediction[0]
            # unscaled state
            unscaled_state = np.array([cap])   
            # unscale action
            bound = self.A * unscaled_state ** self.alpha
            low, high = 0.0001*bound, 0.9999*bound
            unscaled_action = low + (0.5 * (scaled_action + 1.0) * (high - low))
            # compare with optimal action
            opt_action = self.beta * self.alpha * self.A * unscaled_state ** self.alpha
            # absolute deviation 
            dev = abs(unscaled_action - opt_action)
            if cap == 10.0:
                self.rel_dev10 = dev/opt_action
            # deviation as a fraction of optimal solution
            self.rel_dev += dev/opt_action

        return self.rel_dev/10, self.rel_dev10

class EvalProb2():
    def __init__(self):
        self.alpha = 0.8
        self.beta = 0.95
        self.A = 1
        self.cap_upper_bound = 10.0
        self.theta_upper_bound = 1.2
        # relevant states
        self.eval_capital = [10.0, 4.8, 2.66, 1.66, 1.14, 0.85, 0.66, 0.47, 0.31, 0.25]
        self.eval_theta = [0.9, 1, 1.1]
        self.abs_dev = 0
        self.rel_dev = 0

    def evaluate(self, model):
        # model 2: determine mean absolute fractional difference (MAFD) for relevant states
        self.abs_dev = 0
        self.rel_dev = 0
        for cap in self.eval_capital:
            # scale capital
            scaled_cap = np.array([2.0 * (cap / self.cap_upper_bound) - 1.0])
            for theta in self.eval_theta:
                # scale shock (theta)
                scaled_theta = np.array([2.0 * (theta / self.theta_upper_bound) - 1.0])
                # scale state
                scaled_state = np.concatenate((scaled_cap, scaled_theta)).astype(np.float32)
                # predict action 
                prediction = model.predict(np.reshape(np.array([scaled_state]), (1,2)))
                scaled_action = prediction[0]
                # unscaled state
                unscaled_cap = np.array([cap])
                unscaled_theta = np.array([theta])
                unscaled_state = np.concatenate((unscaled_cap, unscaled_theta)).astype(np.float32)
                # unscale action
                bound = self.A * unscaled_state[1] * unscaled_state[0] ** self.alpha
                low, high = 0.0001*bound, 0.9999*bound
                unscaled_action = low + (0.5 * (scaled_action + 1.0) * (high - low))
                # compare with optimal action
                opt_action = self.beta * self.alpha * self.A * unscaled_state[1] * unscaled_state[0] ** self.alpha
                # absolute deviation
                abs_dev = abs(unscaled_action - opt_action)
                # deviation as fraction of optimal solution
                rel_dev = abs_dev/opt_action
                self.abs_dev += abs_dev
                self.rel_dev += rel_dev

        return self.rel_dev/30

class EvalProb3():
    def __init__(self):
        self.rl_reward = []
        self.lq_reward = []
        self.rewards_ratio = []
        self.lq_env = TestProblem3()
        # matrix determining LQ decision-rules
        self.j = np.array([[-0.7883363, 1.3277898, -0.022197655],
                           [0.1501221, 0.2291501, -0.006859907]])
        self.beta = 0.99
        self.cap_lower_bound = 10.0
        self.cap_upper_bound = 11.4
        self.A_lower_bound = 0.5
        self.A_upper_bound = 1.5

    def evaluate(self, model, rl_env):
    # model 3: compare utility achieved by DDPG agent and LQ-benchmark
        self.rl_reward = []
        self.lq_reward = []
        self.rewards_ratio = []
        for i in range(1,6):
            state = rl_env.reset()
            done = False
            rl_score = 0
            nb_steps = 0 
            while not done:
                prediction = model.predict(np.reshape(state, (1,2)), deterministic = True)
                action = prediction[0]
                state, reward, done, info = rl_env.step(action[0])
                rl_score += reward * self.beta ** (nb_steps)
                nb_steps += 1
            
            state = self.lq_env.reset()
            done = False
            lq_score = 0
            nb_steps = 0
            while not done:
                f = np.array([1, state[1], state[0]])
                action = np.matmul(self.j,f)
                state, reward, done, info = self.lq_env.step(action)
                lq_score += reward * self.beta ** (nb_steps)
                nb_steps += 1

            self.rl_reward.append(rl_score)
            self.lq_reward.append(lq_score)
            self.rewards_ratio.append((lq_score-rl_score)/abs(lq_score))

        rl_reward = np.mean(self.rl_reward)
        lq_reward = np.mean(self.lq_reward)
        rewards_ratio = np.mean(self.rewards_ratio)

        return rl_reward, lq_reward, rewards_ratio