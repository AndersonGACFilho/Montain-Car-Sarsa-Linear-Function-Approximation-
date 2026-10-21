import argparse
import os
from typing import List, Optional, Sequence
 
from config import results_dir
from linear_mountaincar.plotting import plot_boxplot, plot_heatmap, plot_learning_curves
from linear_mountaincar.results import (
    CRITERIA, SUMMARY_FIELDS, ConfigKey, ConfigSummary, ResultRepository,
    select_by_keys, select_top_k, summarize, write_csv,
)
 
FIGURES = {
    "boxplot": "boxplot_top5.png",
    "heatmap": "heatmap_median_steps.png",
    "curves": "learning_curves_top5.png",
}
 
 
def make_all(
        out_dir: str = results_dir,
        k: int = 5,
        criterion: str = "median",
        configs: Optional[Sequence[ConfigKey]] = None,
) -> List[ConfigSummary]:
    records = ResultRepository(os.path.join(out_dir, "raw_results.jsonl")).load_all()
    if not records:
        raise FileNotFoundError(f"Nenhum resultado em {out_dir}/. Rode run.py antes.")
 
    summaries = summarize(records)
    selected = (select_by_keys(summaries, configs) if configs
                else select_top_k(summaries, k, criterion))
 
    print(f"{len(records)} runs, {len(summaries)} configurações. Selecionadas:")
    for s in selected:
        print(f"  lr={s.learning_rate:<6} c={s.n_centers:<3} "
              f"sucesso={s.n_success}/{s.n_runs} mediana={s.median:.0f} "
              f"média={s.mean:.0f} IQR={s.iqr:.0f}")
 
    path = lambda name: os.path.join(out_dir, name)  # noqa: E731
    write_csv((s.to_row() for s in summaries), SUMMARY_FIELDS, path("summary_by_config.csv"))
    plot_boxplot(selected, path(FIGURES["boxplot"]))
    plot_heatmap(summaries, path(FIGURES["heatmap"]))
    plot_learning_curves(selected, path(FIGURES["curves"]))
 
    print(f"Gráficos gerados em {out_dir}/")
    return selected
 
 
def _parse_config(text: str) -> ConfigKey:
    lr, centers = text.split(":")
    return float(lr), int(centers)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=results_dir)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--criterion", choices=sorted(CRITERIA), default="median")
    parser.add_argument("--configs", nargs="+", type=_parse_config,
                        help="configurações manuais no formato lr:centros")
    args = parser.parse_args()
    make_all(args.out_dir, args.k, args.criterion, args.configs)
 
 
if __name__ == "__main__":
    main()
