from typing import List

from linear_mountaincar.sarsa_linear_agent import LinearSarsaAgent


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
