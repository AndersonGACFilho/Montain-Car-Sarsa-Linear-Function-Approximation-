import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from linear_mountaincar.experiment import Experiment

ConfigKey = Tuple[float, int]
RunKey = Tuple[float, int, int]

TOTAL_STEPS = 50_000
TARGET_SUCCESSES = 10

@dataclass
class RunRecord:
    learning_rate: float
    n_centers: int
    seed: int
    successes: int
    steps_used: int
    steps_to_success: Optional[int]
    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_successes: Optional[List[bool]] = None

    @property
    def run_key(self) -> RunKey:
        return (self.learning_rate, self.n_centers, self.seed)

    @property
    def config_key(self) -> ConfigKey:
        return (self.learning_rate, self.n_centers)

    @property
    def solved(self) -> bool:
        return self.steps_to_success is not None

    @classmethod
    def from_experiment(
            cls,
            exp: "Experiment",
            total_steps: int = TOTAL_STEPS,
            target: int = TARGET_SUCCESSES,
    ) -> "RunRecord":
        agent = exp.agent
        steps_used = len(exp.step_rewards)
        solved = agent.successes >= target
        return cls(
            learning_rate=float(exp.learning_rate),
            n_centers=int(exp.number_of_centers),
            seed=int(exp.seed),
            successes=int(agent.successes),
            steps_used=int(steps_used),
            steps_to_success=int(steps_used) if solved else None,
            episode_lengths=[int(x) for x in agent.episode_lengths],
            episode_rewards=[float(x) for x in exp.episode_rewards],
            episode_successes=[bool(x) for x in agent.episode_successes],
        )


class ResultRepository:
    def __init__(self, path: str):
        self.path = path

    def append(self, record: RunRecord) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_all(self) -> List[RunRecord]:
        if not os.path.exists(self.path):
            return []
        records: Dict[RunKey, RunRecord] = {}
        with open(self.path, encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = RunRecord(**json.loads(line))
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[aviso] ignorando linha {n} de {self.path}: {e}")
                    continue
                records[rec.run_key] = rec
        return list(records.values())

    def done_keys(self) -> Set[RunKey]:
        return {r.run_key for r in self.load_all()}

@dataclass
class ConfigSummary:
    learning_rate: float
    n_centers: int
    records: List[RunRecord] = field(default_factory=list)
    total_steps: int = TOTAL_STEPS

    @property
    def key(self) -> ConfigKey:
        return (self.learning_rate, self.n_centers)

    @property
    def label(self) -> str:
        return f"lr={self.learning_rate}\nc={self.n_centers}"

    @property
    def steps(self) -> np.ndarray:
        """Passos até 10 sucessos; runs que falharam contam como TOTAL_STEPS
        (valor censurado — o verdadeiro seria >= TOTAL_STEPS)."""
        return np.array([
            r.steps_to_success if r.solved else self.total_steps
            for r in self.records
        ], dtype=float)

    @property
    def n_runs(self) -> int:
        return len(self.records)

    @property
    def n_success(self) -> int:
        return sum(r.solved for r in self.records)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_runs if self.n_runs else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.steps))

    @property
    def mean(self) -> float:
        return float(np.mean(self.steps))

    @property
    def std(self) -> float:
        return float(np.std(self.steps))

    @property
    def iqr(self) -> float:
        q1, q3 = np.percentile(self.steps, [25, 75])
        return float(q3 - q1)

    def to_row(self) -> Dict[str, object]:
        s = self.steps
        return {
            "learning_rate": self.learning_rate,
            "n_centers": self.n_centers,
            "n_runs": self.n_runs,
            "n_success": self.n_success,
            "success_rate": round(self.success_rate, 3),
            "median_steps": self.median,
            "mean_steps": round(self.mean, 1),
            "std_steps": round(self.std, 1),
            "q1_steps": float(np.percentile(s, 25)),
            "q3_steps": float(np.percentile(s, 75)),
            "min_steps": float(s.min()),
            "max_steps": float(s.max()),
        }


SUMMARY_FIELDS = [
    "learning_rate", "n_centers", "n_runs", "n_success", "success_rate",
    "median_steps", "mean_steps", "std_steps", "q1_steps", "q3_steps",
    "min_steps", "max_steps",
]

CRITERIA = {
    "median": lambda s: (s.median, s.mean),
    "mean": lambda s: (s.mean, s.median),
    "success": lambda s: (-s.success_rate, s.median),
    "std": lambda s: (s.std, s.median),
}


def summarize(records: Iterable[RunRecord]) -> List[ConfigSummary]:
    groups: Dict[ConfigKey, ConfigSummary] = {}
    for r in records:
        if r.config_key not in groups:
            groups[r.config_key] = ConfigSummary(*r.config_key)
        groups[r.config_key].records.append(r)
    for g in groups.values():
        g.records.sort(key=lambda r: r.seed)
    return sorted(groups.values(), key=lambda s: s.key)


def select_top_k(summaries: Sequence[ConfigSummary], k: int,
                 criterion: str = "median") -> List[ConfigSummary]:
    return sorted(summaries, key=CRITERIA[criterion])[:k]


def select_by_keys(summaries: Sequence[ConfigSummary],
                   keys: Sequence[ConfigKey]) -> List[ConfigSummary]:
    selected = []
    for lr, c in keys:
        match = [s for s in summaries
                 if np.isclose(s.learning_rate, lr) and s.n_centers == c]
        if not match:
            raise KeyError(f"Configuração lr={lr}, centros={c} não está nos resultados")
        selected.append(match[0])
    return selected


def write_csv(rows: Iterable[Dict[str, object]], fields: Sequence[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)