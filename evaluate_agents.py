from typing import List

from matplotlib import pyplot as plt
import numpy as np

from run_experiments import Experiment, run_experiments


def get_best_k_by_mean_steps_to_success(experiments: List[Experiment], K: int) -> List[int]:
    experiments.sort(key=lambda x: np.mean(
        x.agent.episode_rewards
    ), reverse=True)

def evaluate_agents(experiments: List[Experiment]):
    """
    Evaluate an agent
    """
    get_best_k_by_mean_steps_to_success(
        experiments,
        k = 5
    )
