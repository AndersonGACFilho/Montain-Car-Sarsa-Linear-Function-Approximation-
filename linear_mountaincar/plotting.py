import os
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np

from linear_mountaincar.results import TARGET_SUCCESSES, ConfigSummary, RunRecord

MAX_EPISODE_LEN = 200


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  salvo: {path}")


def plot_boxplot(selected: Sequence[ConfigSummary], path: str) -> None:
    fig, ax = plt.subplots(figsize=(1.8 * len(selected) + 3, 5))
    data = [s.steps for s in selected]
    ax.boxplot(data, showmeans=True, widths=0.5, showfliers=False,
               meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black"))

    rng = np.random.default_rng(0)
    for i, s in enumerate(selected, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=s.n_runs)
        solved = np.array([r.solved for r in s.records])
        ax.scatter(i + jitter[solved], s.steps[solved], s=18, alpha=0.6, color="tab:blue", zorder=3)
        ax.scatter(i + jitter[~solved], s.steps[~solved], s=30, marker="x", color="tab:red", zorder=3)

    ax.set_xticks(range(1, len(selected) + 1))
    ax.set_xticklabels([f"{s.label}\n{s.n_success}/{s.n_runs} ok" for s in selected])
    if any(s.n_success < s.n_runs for s in selected):
        ax.axhline(selected[0].total_steps, ls=":", color="gray", lw=1)
    ax.set_ylabel(f"passos até {TARGET_SUCCESSES} episódios com sucesso")
    ax.set_title("SARSA linear + RBF — configurações selecionadas")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)

def plot_heatmap(summaries: Sequence[ConfigSummary], path: str) -> None:
    lrs = sorted({s.learning_rate for s in summaries})
    centers = sorted({s.n_centers for s in summaries})
    grid = np.full((len(lrs), len(centers)), np.nan)
    rate = np.full_like(grid, np.nan)
    for s in summaries:
        i, j = lrs.index(s.learning_rate), centers.index(s.n_centers)
        grid[i, j], rate[i, j] = s.median, s.success_rate

    fig, ax = plt.subplots(figsize=(1.3 * len(centers) + 2.5, 0.9 * len(lrs) + 2))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="mediana de passos (menor = melhor)")
    ax.set_xticks(range(len(centers)), labels=centers)
    ax.set_yticks(range(len(lrs)), labels=lrs)
    ax.set_xlabel("número de centros RBF")
    ax.set_ylabel("learning rate (α)")
    vmid = np.nanmean(grid)
    for i in range(len(lrs)):
        for j in range(len(centers)):
            if np.isnan(grid[i, j]):
                continue
            color = "white" if grid[i, j] > vmid else "black"
            ax.text(j, i, f"{grid[i, j]:.0f}\n{rate[i, j]:.0%}",
                    ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Mediana de passos e taxa de sucesso por configuração")
    _save(fig, path)

def _episode_successes(record: RunRecord) -> np.ndarray:
    if record.episode_successes is not None:
        return np.asarray(record.episode_successes, dtype=bool)
    return np.asarray(record.episode_lengths) < MAX_EPISODE_LEN


def _cumulative_successes(record: RunRecord, grid: np.ndarray) -> np.ndarray:
    lengths = np.asarray(record.episode_lengths)
    if lengths.size == 0:
        return np.zeros_like(grid, dtype=float)
    ends = np.cumsum(lengths)
    success_ends = ends[_episode_successes(record)]
    return np.searchsorted(success_ends, grid, side="right").astype(float)


def _moving_average(x: List[int], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    n = np.arange(1, x.size + 1)
    lo = np.maximum(0, n - window)
    return (c[n] - c[lo]) / (n - lo)


def _padded(rows: List[List[int]]) -> np.ndarray:
    n = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def plot_learning_curves(selected: Sequence[ConfigSummary], path: str,
                         n_points: int = 500, window: int = 10) -> None:
    total = selected[0].total_steps
    grid = np.linspace(0, total, n_points)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    for s in selected:
        name = s.label.replace("\n", ", ")
        curves = np.array([_cumulative_successes(r, grid) for r in s.records])
        med = np.median(curves, axis=0)
        q1, q3 = np.percentile(curves, [25, 75], axis=0)
        line, = ax1.plot(grid, med, label=name)
        ax1.fill_between(grid, q1, q3, color=line.get_color(), alpha=0.15)

        smooth = [_moving_average(r.episode_lengths, window) for r in s.records]
        lens = _padded(smooth)
        keep = np.sum(~np.isnan(lens), axis=0) >= max(1, s.n_runs // 2)
        with np.errstate(all="ignore"):
            med_len = np.nanmedian(lens, axis=0)
        ax2.plot(np.arange(1, lens.shape[1] + 1)[keep], med_len[keep],
                 color=line.get_color(), label=name)

    ax1.axhline(TARGET_SUCCESSES, ls=":", color="gray")
    ax1.set_xlabel("passos no ambiente")
    ax1.set_ylabel("episódios com sucesso (acumulado)")
    ax1.set_title("Sucessos acumulados (mediana ± IQR)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.axhline(MAX_EPISODE_LEN, ls=":", color="gray")
    ax2.set_xlabel("episódio")
    ax2.set_ylabel("passos no episódio")
    ax2.set_title(f"Duração dos episódios (média móvel {window}, mediana entre seeds)")
    ax2.grid(alpha=0.3)
    _save(fig, path)