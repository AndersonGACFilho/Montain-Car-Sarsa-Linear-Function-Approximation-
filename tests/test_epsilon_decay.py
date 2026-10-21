import os
import sys
import pytest
import numpy as np
import gymnasium as gym

from linear_mountaincar import RBFFeatureExtractor, LinearSarsaAgent

# Make project root importable when running under pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_epsilon_decays_during_train():
    """Run a short training session via agent.train(total_steps) and
    confirm epsilon decreases (but not below epsilon_min).

    If the student hasn't implemented train yet, the test is marked xfail.
    """
    env = gym.make('MountainCar-v0')
    fe = RBFFeatureExtractor(env, n_centers=9, sigma=0.1)

    initial_epsilon = 0.5
    decay = 0.5
    eps_min = 0.1

    agent = LinearSarsaAgent(env=env, feature_extractor=fe,
                             epsilon=initial_epsilon,
                             epsilon_decay=decay,
                             epsilon_min=eps_min)

    try:
        step_rewards, episode_rewards = agent.train(200)
    except NotImplementedError:
        pytest.xfail("agent.train not implemented yet (student exercise)")

    # Epsilon should not increase and should not be below epsilon_min
    assert agent.epsilon <= initial_epsilon + 1e-8
    assert agent.epsilon >= eps_min - 1e-8
