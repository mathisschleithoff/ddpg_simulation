import gym
import numpy as np
import math

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import Schedule

from environments.env_problem1 import OptProblem1
from environments.env_problem2 import OptProblem2
from environments.env_problem3 import OptProblem3
from environments.env_wrapper import TimeLimit2

from evaluation.evaluation_callback import EvalCallback2

env = "env-name"
env = TimeLimit2(env, max_episode_steps = 100)
eval_env_rl = Monitor("test-env-name")
eval_problem_no = "problem-no"
n_actions = env.action_space.shape[-1]
gamma = "gamma"

# schedule for learning rate (linear decay)
def linear_schedule(init_val: float, end_val: float) -> Schedule:

    def func(progress_remaining: float) -> float:
        return init_val - (1 - progress_remaining) * (init_val - end_val)
        
    return func

# model parameters
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
learning_starts = 500
learning_rate = linear_schedule(1e-3,1e-5)
policy_kwargs = dict(net_arch = [64,64])
buffer_size = 10000
replay_buffer_kwargs = dict(handle_timeout_termination = True)

for i in range(1,11):
    # callback for evaluation procedure
    eval_callback = EvalCallback2(eval_env = eval_env_rl, eval_freq=250, n_eval_episodes = 1, deterministic=True, 
                                  render=False, verbose = 1, eval_problem_no = eval_problem_no)
    # callback for saving DDPG model weights
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./', name_prefix='run{}'.format(i))
    callback = CallbackList([eval_callback, checkpoint_callback])                              

    model = DDPG("MlpPolicy", env, verbose=0, train_freq = (25, "step"),
            policy_kwargs=policy_kwargs, gamma = gamma, replay_buffer_kwargs=replay_buffer_kwargs,
            learning_starts = learning_starts, learning_rate = learning_rate, action_noise = action_noise, buffer_size = buffer_size)
    model.learn(total_timesteps=50000, callback = callback) 
    del model

