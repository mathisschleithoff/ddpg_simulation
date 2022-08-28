from gym.wrappers import TimeLimit
import gym
from typing import Optional 

class TimeLimit2(TimeLimit):
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        # set high initial values to prevent erroneous termination
        self.observation_1 = 1000.
        self.observation_2 = 1000.

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.
        Args:
            action: The environment step action
        Returns:
            The environment step ``(observation, reward, done, info)`` with "TimeLimit.truncated"=True
            when truncated (the number of steps elapsed >= max episode steps) or
            "TimeLimit.truncated"=False if the environment terminated
        """
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        self.observation_1 = self.observation_2
        # only capital is considered
        self.observation_2 = observation[0]

        # termination if maximum number of steps is reached or difference in states falls below threshold
        if self._elapsed_steps >= self._max_episode_steps or abs(self.observation_2 - self.observation_1)/abs(self.observation_1) <= 0.01:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.
        Args:
            **kwargs: The kwargs to reset the environment with
        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        self.observation_1 = 1000.
        self.observation_2 = 1000.
        return self.env.reset(**kwargs)
