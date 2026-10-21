
import os
from typing import List, Optional
 
from linear_mountaincar.results import ResultRepository, RunRecord
from .experiment import Experiment
from .run_experiment import run_experiment
 
 
def run_experiments(
        repository: Optional[ResultRepository] = None,
) -> List[Experiment]:
    """
    Runs a set of experiment configurations, with different learning rate,
    number of centers, and seed.
 
    Returns:
        List[Experiment]: experimentos executados NESTA chamada
        (os já gravados em disco são pulados).
    """
    # import tardio: config.py importa o pacote linear_mountaincar, então
    # importar config no topo deste módulo cria um ciclo de imports
    from config import configuration, results_dir
 
    repository = repository or ResultRepository(
        os.path.join(results_dir, "raw_results.jsonl"))
    done = repository.done_keys()
 
    print("Running experiments...")
    print(configuration)
    if done:
        print(f"{len(done)} runs já gravados em {repository.path} serão pulados.")
 
    experiments: List[Experiment] = []
    for learning_rate in configuration.learning_rate_list:
        for number_of_centers in configuration.number_of_centers_list:
            for seed in configuration.seeds:
                if (float(learning_rate), int(number_of_centers), int(seed)) in done:
                    continue
                step_rewards, episode_rewards, agent = run_experiment(
                    learning_rate=learning_rate,
                    n_centers=number_of_centers,
                    seed=seed
                )
                exp = Experiment(
                    learning_rate=learning_rate,
                    number_of_centers=number_of_centers,
                    seed=seed,
                    step_rewards=step_rewards,
                    episode_rewards=episode_rewards,
                    agent=agent
                )
                repository.append(RunRecord.from_experiment(exp))
                experiments.append(exp)
                print(exp)
 
    return experiments
