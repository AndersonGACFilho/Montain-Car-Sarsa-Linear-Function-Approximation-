"""
Script responsible for running different experiment configurations, to
test and compare the different sarsa lambda with rbf linear approximation
agents.
"""

from typing import List, Dict

from run_experiment import run_experiment
from sarsa_linear_agent import LinearSarsaAgent


class Experiment(object):
    learning_rate: float = 0.1
    number_of_centers: int = 9
    seed: int = 42
    max_episodes: int = 10

    agent: LinearSarsaAgent
    step_rewards: List[float]
    episode_rewards: List[float]

    def __init__(
            self,
            learning_rate: float,
            number_of_centers: int,
            step_rewards: List[float],
            episode_rewards: List[float],
            agent: LinearSarsaAgent,
            seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.number_of_centers = number_of_centers
        self.seed = seed

        self.step_rewards = step_rewards
        self.episode_rewards = episode_rewards
        self.agent = agent

    def __str__(self) -> str:
        n_ep = len(self.episode_rewards)
        if n_ep > 0:
            last = self.episode_rewards[-min(10, n_ep):]
            summary = (
                f"episodes={n_ep}, "
                f"best={max(self.episode_rewards):.1f}, "
                f"mean last {len(last)}={sum(last) / len(last):.1f}, "
                f"steps={len(self.step_rewards)}"
            )
        else:
            summary = "no episodes"

        return (
            f"Experiment(lr={self.learning_rate}, "
            f"centers={self.number_of_centers}, "
            f"seed={self.seed} | {summary})"
        )

    __repr__ = __str__

class Configuration(object):
    learning_rate_list: List[float]
    number_of_centers_list: List[int]
    seeds: List[int]
    def __init__(
            self,
            learning_rate_list: List[float],
            number_of_centers_list: List[int],
            seeds_list: List[int]
    ) -> None:
        self.learning_rate_list = learning_rate_list
        self.number_of_centers_list = number_of_centers_list
        self.seeds = seeds_list

    def __str__(self) -> str:
        total = (
                len(self.learning_rate_list)
                * len(self.number_of_centers_list)
                * len(self.seeds)
        )
        return (
            "Configuration(\n"
            f"  learning_rates: {self.learning_rate_list}\n"
            f"  centers:        {self.number_of_centers_list}\n"
            f"  seeds:          {self.seeds}\n"
            f"  total runs:     {total}\n"
            ")"
        )

    __repr__ = __str__

def run_experiments(
) -> List[Experiment]:
    """
    Runs a set of experiment configurations, with different learning rate,
    number of centers, and seed.

    Returns:
        List[Experiment]: List of experiment configurations
    """
    print("Running experiments...")

    experiments: List[Experiment] = []

    configuration: Configuration = Configuration(
        learning_rate_list=[0.005, 0.01, 0.02, 0.05, 0.1],
        number_of_centers_list=[9, 16, 25, 36, 49],
        seeds_list=[42, 1, 8, 256, 1024]
    )

    print(configuration)
    for learning_rate in configuration.learning_rate_list:
        for number_of_centers in configuration.number_of_centers_list:
            for seed in configuration.seeds:
                step_rewards, episode_rewards, agent = run_experiment(
                    learning_rate= learning_rate,
                    n_centers= number_of_centers,
                    seed= seed
                )
                experiments.append(Experiment(
                    learning_rate= learning_rate,
                    number_of_centers= number_of_centers,
                    seed= seed,
                    step_rewards= step_rewards,
                    episode_rewards= episode_rewards,
                    agent= agent
                ))
                print(experiments[-1])
    
    return experiments

if __name__ == "__main__":
    lista_de_experimentos = run_experiments()