import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, EventCallback

# callbacks for evaluation procedure
from evaluation.evaluation_metrics import EvalProb1, EvalProb2, EvalProb3

class EvalCallback2(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        name_prefix: str = "rl_model",
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        eval_problem_no: int = 0
    ):
        super().__init__(verbose = verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        self.eval_problem1 = EvalProb1()
        self.eval_problem2 = EvalProb2()
        self.eval_problem3 = EvalProb3()
        self.eval_problem_no = eval_problem_no

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, name_prefix)
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # Problem 1
        self.evaluations_rel_dev = []
        self.evaluations_rel_dev10 = []
        # Problem 2
        self.evaluations_rel_dev = []
        # Problem 3
        self.evaluations_rl_reward = []
        self.evaluations_lq_reward = []
        self.evaluations_rewards_ratio = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        pass

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            if self.eval_problem_no == 1:
                rel_dev, rel_dev10 = self.eval_problem1.evaluate(self.model, self.eval_env)
            if self.eval_problem_no == 2:
                rel_dev = self.eval_problem2.evaluate(self.model)
            if self.eval_problem_no == 3:
                rl_reward, lq_reward, rewards_ratio = self.eval_problem3.evaluate(self.model, self.eval_env, self.eval_env_lq)

            if self.log_path is not None:
                # Problem 1
                self.evaluations_rel_dev.append(rel_dev)
                self.evaluations_rel_dev10.append(rel_dev10)
                # Problem 2
                #self.evaluations_rel_dev.append(rel_dev)
                # Problem 3
                #self.evaluations_rl_reward.append(rl_reward)
                #self.evaluations_lq_reward.append(lq_reward)
                #self.evaluations_rewards_ratio.append(rewards_ratio)

                np.savez(
                    self.log_path,
                    # Problem 1
                    rel_dev = self.evaluations_rel_dev,
                    rel_dev10 = self.evaluations_rel_dev10,
                    # Problem 2
                    #rel_dev = self.evaluations_rel_dev,
                    # Problem 3
                    #rl_reward = self.evaluations_rl_reward,
                    #lq_reward = self.evaluations_lq_reward,
                    #rewards_ratio = self.evaluations_rewards_ratio,
                )

            if self.verbose > 0:
                # Problem 1, 2
                print(f"Training steps={self.num_timesteps}, " f"MAFD={rel_dev}")
                # Problem 3
                #print(f"Eval num_timesteps={self.num_timesteps}, " f"rl_reward={rl_reward}, " f"lq_reward={lq_reward}")

            # Add to current Logger
            # Problem 1,2
            self.logger.record("eval/rel_dev", rel_dev)
            self.logger.record("eval/rel_dev10", rel_dev10)
            ## Problem 3
            #self.logger.record("eval/rl_reward", float(rl_reward))
            #self.logger.record("eval/lq_reward", float(lq_reward))
            #self.logger.record("eval/rewards_ratio", float(rewards_ratio))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

