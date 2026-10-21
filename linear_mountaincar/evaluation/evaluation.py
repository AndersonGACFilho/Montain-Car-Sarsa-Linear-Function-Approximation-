import numpy as np

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from linear_mountaincar.experiment import Experiment


@dataclass
class ConfigResult:
    learning_rate: float
    number_of_centers: int
    experiments: List[Experiment] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.experiments)

    @property
    def n_success(self) -> int:
        return sum(
            e.agent.successes >= 10
            for e in self.experiments
        )

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_runs if self.n_runs else 0.0

    @property
    def steps(self) -> np.ndarray:
        return np.array([
            len(e.step_rewards) if e.agent.successes >= 10 else 50000
            for e in self.experiments
        ], dtype=float)

    @property
    def rewards(self) -> List[float]:
        rewards = [
            e.agent.episode_rewards
            for e in self.experiments
        ]
        return rewards

    def representative(self) -> Experiment:
        order = np.argsort(self.steps)
        return self.experiments[order[len(order) // 2]]

    def __str__(self) -> str:
        s = self.steps
        return (
            f"Config(lr={self.learning_rate}, centers={self.number_of_centers} | "
            f"sucesso={self.n_success}/{self.n_runs}, "
            f"mediana={float(np.median(s)):.0f}, media={float(np.mean(s)):.0f})"
        )

    __repr__ = __str__

def _mean_episode_rewards(cfg: ConfigResult) -> float:
    return float(np.mean([np.mean(e.episode_rewards) for e in cfg.experiments]))

def _median_episode_rewards(cfg: ConfigResult) -> float:
    return float(np.median([np.median(e.episode_rewards) for e in cfg.experiments]))

def _std_episode_rewards(cfg: ConfigResult) -> float:
    return float(np.std([np.mean(e.episode_rewards) for e in cfg.experiments]))

def _median_steps(cfg: ConfigResult) -> float:
    return float(np.median(cfg.steps))

def _mean_steps(cfg: ConfigResult) -> float:
    return float(np.mean(cfg.steps))

def _std_steps(cfg: ConfigResult) -> float:
    return float(np.std(cfg.steps))

def _success_rate(cfg: ConfigResult) -> float:
    return cfg.success_rate

class EvaluationType(Enum):
    MEDIAN_STEPS = (_median_steps, False)
    MEAN_STEPS = (_mean_steps, False)
    STD_STEPS = (_std_steps, False)
    MEDIAN_REWARDS = (_median_episode_rewards, True)
    MEAN_REWARDS = (_mean_episode_rewards, True)
    STD_REWARDS = (_std_episode_rewards, False)
    SUCCESS_RATE = (_success_rate, True)

    def __init__(self, fn, higher_is_better: bool):
        self.fn = fn
        self.higher_is_better = higher_is_better

    def __call__(self, cfg: ConfigResult) -> float:
        return self.fn(cfg)

    @property
    def dir_name(self) -> str:
        return self.name.lower()

