import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from xml.parsers.expat import model

import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
class StopTrainingOnSuccessRate(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    parent: EvalCallback

    def __init__(self,vec_env: gym.Env, max_no_improvement_evals: int, 
                 success_threshold: float,
                 min_evals: int = 0, 
                 verbose: int = 0,
                 model_name: str = "best_model",
                 model_save_path: Optional[str] = None):
        super().__init__(verbose=verbose)
        self.vec_env = vec_env
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.max_success_evals = 0
        self.no_improvement_evals = 0
        self.success_threshold = success_threshold
        self.best_success_rate = -np.inf
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.threshold_met = 0
    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnSuccessRate`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            success_rate = np.mean(self.parent._is_success_buffer)
            if success_rate < self.success_threshold:
                self.max_success_evals = 0
                if self.threshold_met ==1:
                    self.no_improvement_evals += 1
            else:
                self.max_success_evals += 1
                if success_rate > self.best_success_rate:
                    self.threshold_met =1
                    self.best_success_rate = success_rate
                    self.no_improvement_evals = 0
                    self.model.save(os.path.join(self.model_save_path,self.model_name,self.model_name))
                    stats_path = os.path.join(self.model_save_path,self.model_name ,"vec_normalize.pkl")
                    self.vec_env.save(stats_path)
                    if self.verbose >= 1:
                        print(
                            f"New best success rate: {self.best_success_rate:.2f} - model saved to {os.path.join(self.model_save_path, self.model_name)}"
                        )
                else: 
                    self.no_improvement_evals += 1
            if self.no_improvement_evals >= self.max_no_improvement_evals:
                continue_training = False


        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because max no improvement evaluations was reached for {self.max_no_improvement_evals:d} evaluations"
            )

        return continue_training
