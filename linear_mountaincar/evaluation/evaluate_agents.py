import os
from typing import Dict, List, Optional, Tuple

from linear_mountaincar.evaluation import EvaluationType, ConfigResult
from linear_mountaincar.experiment import Experiment

def group_by_config(experiments: List[Experiment]) -> List[ConfigResult]:
    groups: Dict[Tuple[float, int], ConfigResult] = {}
    for exp in experiments:
        key = (exp.learning_rate, exp.number_of_centers)
        if key not in groups:
            groups[key] = ConfigResult(*key)
        groups[key].experiments.append(exp)
    return list(groups.values())


def get_best_k_by_type(
        configs: List[ConfigResult],
        k: int,
        criterion: EvaluationType = EvaluationType.MEDIAN_STEPS,
) -> List[ConfigResult]:
    return sorted(
        configs, key=criterion, reverse=criterion.higher_is_better
    )[:k]


def save_agents(
        configs: List[ConfigResult],
        out_dir: str,
        criterion: EvaluationType,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for rank, cfg in enumerate(configs, start=1):
        print(f"Rank {rank} | {criterion.name} = {criterion(cfg):.2f} | {cfg}")
        exp = cfg.representative()
        filename = (
            f"rank{rank}_lr{cfg.learning_rate}"
            f"_centers{cfg.number_of_centers}_seed{exp.seed}.pkl"
        )
        exp.agent.save_model(os.path.join(out_dir, filename))


def evaluate_agents(
        experiments: List[Experiment],
        k: int = 5,
        output_dir: Optional[str] = None,
) -> Dict[EvaluationType, List[ConfigResult]]:
    if output_dir is None:
        from config import output_dir  # import tardio (evita ciclo com config.py)
    configs = group_by_config(experiments)
    rankings: Dict[EvaluationType, List[ConfigResult]] = {}
    for criterion in EvaluationType:
        best = get_best_k_by_type(configs, k, criterion)
        save_agents(best, os.path.join(output_dir, criterion.dir_name), criterion)
        rankings[criterion] = best
    return rankings