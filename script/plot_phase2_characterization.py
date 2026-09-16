from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    ROOT
    / "results"
    / "fase_2_caratterizzazione_dimensionale"
    / "campagna_VAS_7_shape_robustness"
)
OUTPUT_DIR = (
    ROOT
    / "plots"
    / "fase_2_caratterizzazione_dimensionale"
    / "campagna_VAS_7_shape_robustness"
)

SIZES = [4096, 8192, 16384]
K_VALUES = [3, 6, 8, 20, 32]
SHAPES = ["square", "n2m", "m3n"]
BACKENDS = ["cuda_naive", "cuda_warp_column", "cuda_warp_smem", "cublas"]

BACKEND_LABELS = {
    "cuda_naive": "CUDA naive",
    "cuda_warp_column": "CUDA warp column-major",
    "cuda_warp_smem": "CUDA warp shared-memory",
    "cublas": "cuBLAS",
}
BACKEND_COLORS = {
    "cuda_naive": "#4C78A8",
    "cuda_warp_column": "#D55E00",
    "cuda_warp_smem": "#009E73",
    "cublas": "#6F6F6F",
}
SHAPE_LABELS = {
    "square": r"$M=N$",
    "n2m": r"$N=2M$",
    "m3n": r"$M=3N$",
}
SHAPE_COLORS = {
    "n2m": "#56B4E9",
    "m3n": "#CC79A7",
}
COLOR_GRID = "#B8B8B8"
COLOR_REFERENCE = "#3F3F3F"

FILE_PATTERN = re.compile(
    r"^campagna_VAS_7_s(?P<size>4096|8192|16384)_"
    r"(?P<shape>square|n2m|m3n)_"
    r"(?P<backend>cuda_naive|cuda_warp_column_xcolumn|cuda_warp_smem|cublas)\.csv$"
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

def canonical_backend(name: str) -> str:
    if name == "cuda_warp_column_xcolumn":
        return "cuda_warp_column"
    return name

def load_results() -> pd.DataFrame:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Cartella non trovata: {INPUT_DIR}")

    frames: list[pd.DataFrame] = []
    matched_files: list[Path] = []
    required = {
        "scalar",
        "M",
        "N",
        "k",
        "P",
        "reps",
        "t_kernel_mean_s",
        "t_kernel_std_s",
        "t_kernel_cv_pct",
        "gflops_kernel",
    }

    for csv_path in sorted(INPUT_DIR.glob("*.csv")):
        match = FILE_PATTERN.match(csv_path.name)
        if match is None:
            continue

        df = pd.read_csv(csv_path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name}: colonne mancanti: {sorted(missing)}")

        numeric_columns = [
            "M",
            "N",
            "k",
            "P",
            "reps",
            "t_kernel_mean_s",
            "t_kernel_std_s",
            "t_kernel_cv_pct",
            "gflops_kernel",
        ]
        df = df.copy()
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid = df[numeric_columns].isna().any(axis=1)
        if invalid.any():
            rows = (df.index[invalid] + 2).tolist()
            raise ValueError(f"{csv_path.name}: valori non validi alle righe {rows}")

        df["size"] = int(match.group("size"))
        df["shape"] = match.group("shape")
        df["backend"] = canonical_backend(match.group("backend"))
        df["source_file"] = csv_path.name

        if set(df["k"].astype(int)) != set(K_VALUES):
            raise ValueError(
                f"{csv_path.name}: k attesi {K_VALUES}, "
                f"trovati {sorted(df['k'].astype(int).unique())}"
            )
        if df.duplicated(subset=["k"]).any():
            raise ValueError(f"{csv_path.name}: risultati duplicati per k")
        if (df["t_kernel_mean_s"] <= 0).any() or (df["gflops_kernel"] <= 0).any():
            raise ValueError(f"{csv_path.name}: tempi e throughput devono essere positivi")

        expected_gflops = (
            2.0 * df["M"] * df["N"] * df["k"]
            / df["t_kernel_mean_s"]
            / 1.0e9
        )
        if not np.allclose(
            df["gflops_kernel"], expected_gflops, rtol=1.0e-5, atol=1.0e-6
        ):
            raise ValueError(
                f"{csv_path.name}: gflops_kernel non coerente con "
                "2*M*N*k/t_kernel_mean_s"
            )

        df["gflops_kernel_std_approx"] = (
            df["gflops_kernel"]
            * df["t_kernel_std_s"]
            / df["t_kernel_mean_s"]
        )
        frames.append(df)
        matched_files.append(csv_path)

    expected_file_count = len(SIZES) * len(SHAPES) * len(BACKENDS)
    if len(matched_files) != expected_file_count:
        raise ValueError(
            f"Attesi {expected_file_count} CSV aggregati, trovati {len(matched_files)}"
        )

    results = pd.concat(frames, ignore_index=True)
    results["k"] = results["k"].astype(int)
    results["size"] = results["size"].astype(int)

    keys = ["size", "shape", "backend", "k"]
    if results.duplicated(subset=keys).any():
        duplicated = results.loc[results.duplicated(subset=keys, keep=False), keys]
        raise ValueError(f"Configurazioni duplicate:\n{duplicated.to_string(index=False)}")

    expected_rows = expected_file_count * len(K_VALUES)
    if len(results) != expected_rows:
        raise ValueError(f"Attese {expected_rows} righe, trovate {len(results)}")
    if set(results["scalar"].astype(str)) != {"double"}:
        raise ValueError("La campagna deve essere interamente in precisione double")
    if set(results["P"].astype(int)) != {1}:
        raise ValueError("La campagna deve essere interamente con P=1")

    return results

def save_figure(fig: plt.Figure, basename: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{basename}.png"
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path

def add_bar_labels(ax, bars, values, fontsize: float = 7.8) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.0f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=bar.get_facecolor(),
        )

def plot_square_kernel_comparison(results: pd.DataFrame) -> tuple[Path, Path]:
    square = results[results["shape"] == "square"].copy()
    x = np.arange(len(K_VALUES), dtype=float)
    width = 0.19

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.8), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.78, wspace=0.10)
    fig.suptitle(
        "Fase 2: confronto dei kernel su matrici quadrate",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "FP64, P=1 · throughput del kernel basato sul tempo medio · barre d'errore: deviazione standard approssimata",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    max_value = float(
        (square["gflops_kernel"] + square["gflops_kernel_std_approx"]).max()
    )
    for axis_index, (ax, size) in enumerate(zip(axes, SIZES)):
        panel = square[square["size"] == size]
        for backend_index, backend in enumerate(BACKENDS):
            subset = (
                panel[panel["backend"] == backend]
                .set_index("k")
                .reindex(K_VALUES)
            )
            offset = (backend_index - (len(BACKENDS) - 1) / 2.0) * width
            values = subset["gflops_kernel"].to_numpy()
            errors = subset["gflops_kernel_std_approx"].to_numpy()
            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=2.0,
                color=BACKEND_COLORS[backend],
                edgecolor="white",
                linewidth=0.35,
                label=BACKEND_LABELS[backend],
                error_kw={"elinewidth": 0.75, "alpha": 0.75},
            )
            if axis_index == len(axes) - 1:
                add_bar_labels(ax, bars, values)

        ax.set_title(rf"$M=N={size}$")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("Ampiezza del multivettore k")
        ax.set_ylim(0.0, max_value * 1.15)
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.50,
            color=COLOR_GRID,
        )
    axes[0].set_ylabel("Throughput del kernel [GFLOP/s]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.055,
        "La scala verticale è identica nei tre pannelli; i valori numerici sono riportati sul pannello S=16384.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    return save_figure(fig, "campagna_VAS_7_square_kernel_comparison_bars")

def plot_size_scaling(results: pd.DataFrame) -> tuple[Path, Path]:
    square = results[results["shape"] == "square"].copy()
    selected_k = [3, 8, 32]
    x = np.arange(len(SIZES), dtype=float)
    width = 0.19

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.7), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.78, wspace=0.11)
    fig.suptitle(
        "Fase 2: effetto della crescita della matrice",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        r"Matrici quadrate, FP64, P=1 · $S=M=N$ · valori rappresentativi di k",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    max_value = float(square["gflops_kernel"].max())
    for axis_index, (ax, k_value) in enumerate(zip(axes, selected_k)):
        panel = square[square["k"] == k_value]
        for backend_index, backend in enumerate(BACKENDS):
            subset = (
                panel[panel["backend"] == backend]
                .set_index("size")
                .reindex(SIZES)
            )
            offset = (backend_index - (len(BACKENDS) - 1) / 2.0) * width
            values = subset["gflops_kernel"].to_numpy()
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=BACKEND_COLORS[backend],
                edgecolor="white",
                linewidth=0.35,
                label=BACKEND_LABELS[backend],
            )
            add_bar_labels(ax, bars, values, fontsize=7.5)

        ax.set_title(rf"$k={k_value}$")
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in SIZES])
        ax.set_xlabel(r"Dimensione nominale $S$")
        ax.set_ylim(0.0, max_value * 1.18)
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.50,
            color=COLOR_GRID,
        )
    axes[0].set_ylabel("Throughput del kernel [GFLOP/s]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.055,
        "I pannelli k=3, 8 e 32 rappresentano rispettivamente multivettori piccoli, intermedi e grandi.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    return save_figure(fig, "campagna_VAS_7_size_scaling_bars")

def build_shape_ratios(results: pd.DataFrame) -> pd.DataFrame:
    square = (
        results[results["shape"] == "square"]
        [["size", "backend", "k", "gflops_kernel"]]
        .rename(columns={"gflops_kernel": "gflops_square"})
    )
    rectangular = results[results["shape"].isin(["n2m", "m3n"])].copy()
    ratios = rectangular.merge(
        square,
        on=["size", "backend", "k"],
        how="left",
        validate="many_to_one",
    )
    if ratios["gflops_square"].isna().any():
        raise ValueError("Manca il riferimento quadrato per alcune configurazioni")
    ratios["relative_to_square_pct"] = (
        100.0 * ratios["gflops_kernel"] / ratios["gflops_square"]
    )
    return ratios

def plot_shape_robustness(
    ratios: pd.DataFrame,
    size: int,
) -> tuple[Path, Path]:
    panel = ratios[ratios["size"] == size]
    x = np.arange(len(K_VALUES), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 6.5), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.76, wspace=0.10)
    fig.suptitle(
        f"Fase 2: robustezza rispetto alla forma della matrice (S={size})",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.885,
        r"Variazione rispetto alla matrice quadrata con lo stesso kernel, k e S · quadrata = 0%",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    panel = panel.copy()
    panel["delta_vs_square_pct"] = panel["relative_to_square_pct"] - 100.0
    max_abs_delta = float(panel["delta_vs_square_pct"].abs().max())
    y_limit = max(10.0, np.ceil((max_abs_delta + 2.0) / 5.0) * 5.0)

    for ax, backend in zip(axes, BACKENDS):
        backend_data = panel[panel["backend"] == backend]
        for shape_index, shape in enumerate(["n2m", "m3n"]):
            subset = (
                backend_data[backend_data["shape"] == shape]
                .set_index("k")
                .reindex(K_VALUES)
            )
            offset = (shape_index - 0.5) * width
            values = subset["delta_vs_square_pct"].to_numpy()
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=SHAPE_COLORS[shape],
                edgecolor="white",
                linewidth=0.4,
                label=SHAPE_LABELS[shape],
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:+.1f}",
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, 3 if value >= 0.0 else -4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    fontsize=7.6,
                    color="#333333",
                )

        ax.axhspan(-5.0, 5.0, color="#E6E6E6", alpha=0.55, zorder=0)
        ax.axhline(
            0.0,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )
        ax.set_title(BACKEND_LABELS[backend])
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("k")
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.65,
            alpha=0.45,
            color=COLOR_GRID,
        )
    axes[0].set_ylabel("Variazione rispetto al caso quadrato [%]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.825),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.055,
        "La fascia grigia indica uno scostamento entro ±5%; valori positivi indicano un vantaggio rispetto al caso quadrato.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    return save_figure(fig, f"campagna_VAS_7_shape_robustness_s{size}_bars")

def main() -> None:
    results = load_results()
    ratios = build_shape_ratios(results)

    generated: list[Path] = []
    generated.extend(plot_square_kernel_comparison(results))
    generated.extend(plot_size_scaling(results))
    for size in SIZES:
        generated.extend(plot_shape_robustness(ratios, size))

    print("Grafici a istogramma della Fase 2")
    print(f"CSV aggregati letti: {results['source_file'].nunique()}")
    print(f"Configurazioni analizzate: {len(results)}")
    for path in generated:
        print(path)

if __name__ == "__main__":
    main()
