"""
Script responsible for running the experiment, with different learning rate,
number of centers, seed and max episodes.
"""
import os
import random
from typing import Tuple, List

import numpy as np
import gymnasium as gym

from linear_mountaincar.sarsa_linear_agent import LinearSarsaAgent
from linear_mountaincar.feature_extractors import RBFFeatureExtractor


def run_experiment(
        steps: int = 50000,
        learning_rate:float = 0.01,
        n_centers:int = 9,
        seed: int = 42,
        max_episodes: int = 10
) -> Tuple[List[float],List[float], LinearSarsaAgent]:
    """
    This method is responsible for running the experiment, with
    different learning rate, number of centers, seed and max episodes.
    To be easier to compare with other parameters configurations

    Args
        steps: number of steps to run the experiment
        learning_rate: learning rate for the agent
        n_centers: number of centers to use
        seed: random seed
        max_episodes: maximum number of episodes to run

    Returns
        Tuple[list[float],list[float], LinearSarsaAgent]: the list of rewards,
            The list of accumulated rewards per episode,
            The agent that was trained on
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make('MountainCar-v0')

    env.reset(seed=seed)
    env.action_space.seed(seed)

    env = gym.make('MountainCar-v0')

    feature_extractor = RBFFeatureExtractor(
        env=env, n_centers=n_centers, sigma=0.2)

    agent = LinearSarsaAgent(
        env=env,
        seed=seed,
        feature_extractor=feature_extractor,
        learning_rate=learning_rate,
        epsilon=0.1,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        discount_factor=1.0
    )

    print(f"Feature vector size: {feature_extractor.n_features}")
    print(f"Number of actions: {env.action_space.n}")
    # pylint: disable=redefined-outer-name
    step_rewards = []
    episode_rewards = []
    try:
        step_rewards, episode_rewards = (
            agent.train(
                steps,
                max_episodes
            )
        )
        print(f"Completed training: steps={len(step_rewards)}, episodes={len(episode_rewards)}")
    except NotImplementedError:
        print("agent.train() not implemented: implement train(total_steps) to run the demo.")
    except Exception as e:
        print(f"agent.train() raised {type(e).__name__}: {e}")

    env.close()

    return step_rewards, episode_rewards, agent