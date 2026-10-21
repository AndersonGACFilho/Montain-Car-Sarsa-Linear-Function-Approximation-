from linear_mountaincar.experiment import run_experiments
from linear_mountaincar.evaluation import evaluate_agents
from plot import make_all

if __name__ == "__main__":
    experimentos = run_experiments()
    if experimentos:
        evaluate_agents(experimentos)
    make_all()
